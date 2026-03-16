"""
Optional Challenge — Full Pipeline Entry Point

Processes 10 sequential LiDAR frames through:
    1. Ground removal        (RANSAC — open3d)
    2. Instance clustering   (DBSCAN — scikit-learn)
    3. Cluster classification (PointNet++ — checkpoints/best_model.pth, MPS)
    4. Multi-object tracking  (SORT — filterpy + scipy)

All results are written to optional_challenge/results/:
    segmentation/frame_XX.npz     — labelled point clouds
    tracking/frame_XX.json        — per-frame tracking output
    visualization/frame_XX.png    — BEV plots with track overlays
    visualization/trajectories.png — full trajectory summary
    visualization/class_distribution.png

Usage:
    cd optional_challenge/
    python run_pipeline.py
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow importing sibling modules regardless of CWD
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from ground_removal import segment_ground
from clustering import cluster_objects, extract_clusters
from classifier import ClusterClassifier
from tracker import MultiClassTracker, FrameDetection
from visualize import (
    save_bev_frame,
    save_trajectory_plot,
    save_class_distribution_plot,
    save_dashboard,
    save_animated_gif,
)

# ===========================================================================
# GLOBAL CONFIG — edit these to tune the pipeline
# ===========================================================================

# --- Paths ------------------------------------------------------------------
_REPO_ROOT         = _THIS_DIR.parent
DATA_DIR           = _REPO_ROOT / "data" / "optional_challenge_data"
CHECKPOINT_PATH    = _REPO_ROOT / "checkpoints" / "best_model.pth"
RESULTS_DIR        = _THIS_DIR / "results"

# --- Device -----------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[warn] MPS/CUDA not available — falling back to CPU.")
    return torch.device("cpu")

# --- Ground removal ---------------------------------------------------------
GROUND_DIST_THRESHOLD = 0.25   # metres; points within this of the RANSAC plane are ground
GROUND_MAX_Z          = 0.5    # seed cloud height cap (avoids fitting walls)

# --- DBSCAN clustering ------------------------------------------------------
DBSCAN_EPS         = 0.6   # neighbourhood radius (m)
DBSCAN_MIN_SAMPLES = 5     # core-point threshold

# --- Classification ---------------------------------------------------------
# "background" clusters are not passed to the tracker (they are not objects
# of interest) but ARE labelled in the per-frame segmentation output
BACKGROUND_CLASS = "background"

# --- Tracker ----------------------------------------------------------------
TRACKER_MAX_AGE  = 3   # frames before an unmatched track is removed
TRACKER_MIN_HITS = 1   # minimum matches before a track is reported

# --- Visualization ----------------------------------------------------------
BEV_XLIM       = (-50.0, 50.0)   # metres
BEV_YLIM       = (-50.0, 50.0)
TRAIL_LENGTH   = 5                # how many past positions to keep per track
GIF_DURATION_MS = 600             # ms per frame in the animated GIF

# ===========================================================================
# End of config
# ===========================================================================


# ---------------------------------------------------------------------------
# Label constants (must stay consistent with visualize.py)
# ---------------------------------------------------------------------------
LABEL_GROUND      = 0
LABEL_BACKGROUND  = 1
LABEL_CAR         = 2
LABEL_PEDESTRIAN  = 3
LABEL_BICYCLIST   = 4

CLASS_TO_LABEL: dict[str, int] = {
    "background": LABEL_BACKGROUND,
    "car":        LABEL_CAR,
    "pedestrian": LABEL_PEDESTRIAN,
    "bicyclist":  LABEL_BICYCLIST,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frame(path: Path) -> np.ndarray:
    """Load a binary frame → (N, 5) float32: x, y, z, intensity, ring."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


def _np_to_list(arr) -> list:
    """Recursively convert numpy scalars / arrays to plain Python for JSON."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (np.floating, np.integer)):
        return arr.item()
    return arr


def tracked_objects_to_json(tracked, frame_idx: int) -> list[dict]:
    return [
        {
            "frame":       frame_idx,
            "track_id":    obj.track_id,
            "class":       obj.class_name,
            "centroid":    _np_to_list(obj.centroid),
            "bbox_min":    _np_to_list(obj.bbox_min),
            "bbox_max":    _np_to_list(obj.bbox_max),
            "age":         obj.age,
            "hits":        obj.hits,
            "confidence":  float(obj.confidence),
        }
        for obj in tracked
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    device = get_device()
    print(f"Device: {device}")

    seg_dir  = RESULTS_DIR / "segmentation"
    trk_dir  = RESULTS_DIR / "tracking"
    vis_dir  = RESULTS_DIR / "visualization"
    for d in (seg_dir, trk_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Discover frames (sorted by timestamp embedded in filename)
    frame_paths = sorted(DATA_DIR.glob("*.bin"))
    if not frame_paths:
        raise FileNotFoundError(f"No .bin files found in {DATA_DIR}")
    print(f"Found {len(frame_paths)} frame(s) in {DATA_DIR}")

    # Load classifier
    print(f"Loading classifier from {CHECKPOINT_PATH} …")
    classifier = ClusterClassifier(str(CHECKPOINT_PATH), device)

    # Initialise tracker (reset ID counter for reproducibility)
    MultiClassTracker.reset_ids()
    tracker = MultiClassTracker(
        max_age=TRACKER_MAX_AGE,
        min_hits=TRACKER_MIN_HITS,
    )

    # Accumulators for summary plots and trail rendering
    all_tracked_per_frame: list = []
    per_frame_class_counts: list[dict[str, int]] = []
    all_tracking_json: list[dict] = []
    # trail_history: {track_id: [(x, y), ...]} — last TRAIL_LENGTH positions
    trail_history: dict[int, list[tuple[float, float]]] = {}

    # -----------------------------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------------------------
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Frames")):
        t_frame = time.time()

        # 1. Load ---------------------------------------------------------------
        points = load_frame(frame_path)   # (N, 5)
        xyz    = points[:, :3]            # (N, 3)
        N      = len(points)

        # 2. Ground removal -----------------------------------------------------
        ground_mask = segment_ground(
            xyz,
            distance_threshold=GROUND_DIST_THRESHOLD,
            max_ground_z=GROUND_MAX_Z,
        )
        obj_mask  = ~ground_mask
        xyz_obj   = xyz[obj_mask]
        pts_obj   = points[obj_mask]

        # 3. DBSCAN clustering --------------------------------------------------
        cluster_labels = cluster_objects(
            xyz_obj,
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
        )
        clusters = extract_clusters(xyz_obj, pts_obj, cluster_labels)

        # 4. Classify clusters --------------------------------------------------
        results = classifier.classify_batch(clusters)

        # 5. Build per-point label array ----------------------------------------
        labels = np.zeros(N, dtype=np.int32)           # default: ground
        labels[obj_mask] = LABEL_BACKGROUND             # non-ground default

        # Map DBSCAN cluster IDs back to the original (full-cloud) index space
        orig_indices = np.where(obj_mask)[0]
        for clust, res in zip(clusters, results):
            lbl = CLASS_TO_LABEL.get(res.class_name, LABEL_BACKGROUND)
            cl_member_mask  = cluster_labels == clust.cluster_id
            orig_indices_cl = orig_indices[cl_member_mask]
            labels[orig_indices_cl] = lbl

        # 6. Class-point counts (for distribution plot) -------------------------
        label_names  = {
            LABEL_GROUND:     "ground",
            LABEL_BACKGROUND: "background",
            LABEL_CAR:        "car",
            LABEL_PEDESTRIAN: "pedestrian",
            LABEL_BICYCLIST:  "bicyclist",
        }
        frame_counts = {
            name: int((labels == lbl_idx).sum())
            for lbl_idx, name in label_names.items()
        }
        per_frame_class_counts.append(frame_counts)

        # 7. Prepare detections for tracker ------------------------------------
        frame_detections: list[FrameDetection] = []
        for clust, res in zip(clusters, results):
            if res.class_name == BACKGROUND_CLASS:
                continue
            frame_detections.append(
                FrameDetection(
                    class_name=res.class_name,
                    centroid=clust.centroid,
                    bbox_min=clust.bbox_min,
                    bbox_max=clust.bbox_max,
                    confidence=res.confidence,
                )
            )

        # 8. Update tracker ----------------------------------------------------
        tracked = tracker.update(frame_detections)
        all_tracked_per_frame.append(tracked)

        # 9. Save segmentation result ------------------------------------------
        np.savez_compressed(
            str(seg_dir / f"frame_{frame_idx:02d}.npz"),
            points=points,
            labels=labels,
        )

        # 10. Save tracking result ---------------------------------------------
        frame_json = tracked_objects_to_json(tracked, frame_idx)
        all_tracking_json.extend(frame_json)
        with open(str(trk_dir / f"frame_{frame_idx:02d}.json"), "w") as fh:
            json.dump(frame_json, fh, indent=2)

        # 11. Update trail history (before BEV render) -------------------------
        for obj in tracked:
            tid = obj.track_id
            if tid not in trail_history:
                trail_history[tid] = []
            trail_history[tid].append((float(obj.centroid[0]), float(obj.centroid[1])))
            if len(trail_history[tid]) > TRAIL_LENGTH:
                trail_history[tid] = trail_history[tid][-TRAIL_LENGTH:]

        # 12. Save BEV visualization -------------------------------------------
        save_bev_frame(
            points=points,
            labels=labels,
            tracked_objects=tracked,
            frame_idx=frame_idx,
            output_path=str(vis_dir / f"frame_{frame_idx:02d}.png"),
            trail_history=trail_history,
            xlim=BEV_XLIM,
            ylim=BEV_YLIM,
        )

        # 13. Per-frame console summary ----------------------------------------
        n_clusters   = len(clusters)
        cls_cnt      = Counter(r.class_name for r in results)
        n_tracked    = len(tracked)
        elapsed_ms   = (time.time() - t_frame) * 1000
        print(
            f"  frame {frame_idx:02d} | pts={N:,}  ground={frame_counts['ground']:,}"
            f"  clusters={n_clusters}"
            f"  [car={cls_cnt.get('car',0)}"
            f"  ped={cls_cnt.get('pedestrian',0)}"
            f"  bike={cls_cnt.get('bicyclist',0)}"
            f"  bg={cls_cnt.get('background',0)}]"
            f"  tracks={n_tracked}"
            f"  {elapsed_ms:.0f} ms"
        )

    # -----------------------------------------------------------------------
    # Summary outputs
    # -----------------------------------------------------------------------
    print("\nSaving summary outputs …")

    # Consolidated tracking JSON
    with open(str(trk_dir / "all_frames.json"), "w") as fh:
        json.dump(all_tracking_json, fh, indent=2)

    # Trajectory plot
    save_trajectory_plot(
        all_tracked_per_frame,
        output_path=str(vis_dir / "trajectories.png"),
        xlim=BEV_XLIM,
        ylim=BEV_YLIM,
    )

    # Class distribution plot
    save_class_distribution_plot(
        per_frame_class_counts,
        output_path=str(vis_dir / "class_distribution.png"),
    )

    # Dashboard
    save_dashboard(
        all_tracked_per_frame,
        per_frame_class_counts,
        output_path=str(vis_dir / "dashboard.png"),
    )

    # Animated GIF (requires Pillow — skipped gracefully if not installed)
    frame_pngs = [str(vis_dir / f"frame_{i:02d}.png") for i in range(len(frame_paths))]
    save_animated_gif(
        frame_pngs,
        output_path=str(vis_dir / "tracking.gif"),
        duration_ms=GIF_DURATION_MS,
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_tracks = set()
    for frame_tracks in all_tracked_per_frame:
        for obj in frame_tracks:
            total_tracks.add((obj.class_name, obj.track_id))

    track_by_class: Counter = Counter(cls for cls, _ in total_tracks)

    print("\n" + "=" * 60)
    print(f"Pipeline complete in {(time.time() - t_total):.1f} s")
    print(f"Frames processed   : {len(frame_paths)}")
    print(f"Unique track IDs   : {len(total_tracks)}")
    for cls_name in ("car", "pedestrian", "bicyclist"):
        print(f"  {cls_name:<12}: {track_by_class.get(cls_name, 0)}")
    print("\nResults written to:")
    print(f"  Segmentation  → {seg_dir}")
    print(f"  Tracking      → {trk_dir}")
    print(f"  Visualization → {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
