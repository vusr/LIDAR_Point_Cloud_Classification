"""
Optional Challenge — Improved Pipeline Entry Point

An enhanced version of run_pipeline.py that addresses poor pedestrian /
cyclist segmentation through several targeted modifications:

    1. Tighter ground removal   — reduced RANSAC threshold (0.15 m) to
       preserve pedestrian feet.
    2. Two-pass DBSCAN          — a fine pass (eps=0.35) to capture small,
       dense objects (pedestrians / cyclists), followed by a coarse pass
       (eps=1.2) on residual noise to recover cars and larger objects.
    3. Lower minimum cluster    — MIN_CLUSTER_POINTS=5 so distant
       pedestrians are not silently discarded.
    4. Geometric post-filtering — bounding-box dimensions are checked
       after classification to veto implausible assignments (e.g.
       a 10 m-wide cluster labelled "pedestrian") and to re-cluster
       oversized pedestrian blobs with a tighter eps.
    5. Cluster classification   — PointNet++ (same checkpoint as before).
    6. Multi-object tracking    — SORT with Kalman filter (unchanged).

Results are written to  optional_challenge/results_improved/  so they do
not overwrite results from run_pipeline.py or run_pipeline_salsanext.py.

Usage:
    cd optional_challenge/
    python run_pipeline_improved.py
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
from clustering import cluster_objects, extract_clusters, Cluster
from classifier import ClusterClassifier, ClassifyResult
from tracker import MultiClassTracker, FrameDetection
from visualize import (
    save_bev_frame,
    save_trajectory_plot,
    save_class_distribution_plot,
    save_dashboard,
    save_animated_gif,
)

# ===========================================================================
# GLOBAL CONFIG
# ===========================================================================

# --- Paths ------------------------------------------------------------------
_REPO_ROOT         = _THIS_DIR.parent
DATA_DIR           = _REPO_ROOT / "data" / "optional_challenge_data"
CHECKPOINT_PATH    = _REPO_ROOT / "checkpoints" / "best_model.pth"
RESULTS_DIR        = _THIS_DIR / "results_improved"

# --- Device -----------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[warn] MPS/CUDA not available — falling back to CPU.")
    return torch.device("cpu")

# --- Ground removal (tighter threshold to preserve pedestrian feet) ---------
GROUND_DIST_THRESHOLD = 0.15
GROUND_MAX_Z          = 0.5

# --- Two-pass DBSCAN -------------------------------------------------------
# Pass 1 (fine): catches pedestrians, cyclists, small objects
FINE_EPS          = 0.35
FINE_MIN_SAMPLES  = 3

# Pass 2 (coarse): recovers cars and large objects from pass-1 residuals
COARSE_EPS         = 1.4
COARSE_MIN_SAMPLES = 6

# --- Cluster size limits ----------------------------------------------------
MIN_CLUSTER_POINTS = 5
MAX_CLUSTER_POINTS = 8000
MAX_BBOX_DIAGONAL  = 20.0

# --- Geometric plausibility bounds per class --------------------------------
GEOMETRY_BOUNDS: dict[str, dict] = {
    "pedestrian": {"max_diag": 3.0, "max_width": 1.5, "max_length": 1.5, "max_height": 2.5},
    "bicyclist":  {"max_diag": 4.0, "max_width": 2.0, "max_length": 3.0, "max_height": 2.5},
    "car":        {"max_diag": 15.0, "max_width": 7.0, "max_length": 12.0, "max_height": 7.0},
}

# Oversized pedestrian clusters are re-split with this eps
RESPLIT_EPS = 0.25
RESPLIT_MIN_SAMPLES = 3

# --- Classification ---------------------------------------------------------
BACKGROUND_CLASS = "background"

# --- Tracker ----------------------------------------------------------------
TRACKER_MAX_AGE  = 3
TRACKER_MIN_HITS = 1

# --- Visualization ----------------------------------------------------------
BEV_XLIM       = (-50.0, 50.0)
BEV_YLIM       = (-50.0, 50.0)
TRAIL_LENGTH   = 5
GIF_DURATION_MS = 600

# ===========================================================================
# End of config
# ===========================================================================


# ---------------------------------------------------------------------------
# Label constants
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
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


def _np_to_list(arr) -> list:
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
# Two-pass clustering
# ---------------------------------------------------------------------------

def two_pass_cluster(
    xyz: np.ndarray,
    pts_full: np.ndarray,
) -> list[Cluster]:
    """
    Two-pass DBSCAN to simultaneously capture small dense objects
    (pedestrians, cyclists) and large sparse objects (cars).

    Pass 1 — fine eps:   finds tight pedestrian-scale clusters.
    Pass 2 — coarse eps: runs on the noise left over from pass 1
                         to recover cars and larger structures.

    Returns a single merged list of Cluster objects with unique IDs.
    """
    if len(xyz) == 0:
        return []

    # --- Pass 1 (fine) -------------------------------------------------------
    labels_fine = cluster_objects(xyz, eps=FINE_EPS, min_samples=FINE_MIN_SAMPLES)
    clusters_fine = extract_clusters(
        xyz, pts_full, labels_fine,
        min_points=MIN_CLUSTER_POINTS,
        max_points=MAX_CLUSTER_POINTS,
        max_diagonal=MAX_BBOX_DIAGONAL,
    )

    # --- Pass 2 (coarse) on residual noise -----------------------------------
    # Points that were not assigned to any valid cluster in pass 1
    assigned_ids = {c.cluster_id for c in clusters_fine}
    assigned_mask = np.zeros(len(labels_fine), dtype=bool)
    for cid in assigned_ids:
        assigned_mask |= (labels_fine == cid)

    residual_mask = ~assigned_mask
    residual_xyz  = xyz[residual_mask]
    residual_full = pts_full[residual_mask]

    clusters_coarse: list[Cluster] = []
    if len(residual_xyz) > 0:
        labels_coarse = cluster_objects(
            residual_xyz, eps=COARSE_EPS, min_samples=COARSE_MIN_SAMPLES,
        )
        clusters_coarse = extract_clusters(
            residual_xyz, residual_full, labels_coarse,
            min_points=MIN_CLUSTER_POINTS,
            max_points=MAX_CLUSTER_POINTS,
            max_diagonal=MAX_BBOX_DIAGONAL,
        )

    # --- Merge with unique IDs -----------------------------------------------
    max_id = max((c.cluster_id for c in clusters_fine), default=-1)
    for c in clusters_coarse:
        max_id += 1
        c.cluster_id = max_id

    all_clusters = clusters_fine + clusters_coarse
    return all_clusters


# ---------------------------------------------------------------------------
# Geometric plausibility check & re-splitting
# ---------------------------------------------------------------------------

def _bbox_dims(cluster: Cluster) -> tuple[float, float, float]:
    """Return (width_x, length_y, height_z) of a cluster's bbox."""
    size = cluster.bbox_max - cluster.bbox_min
    return float(size[0]), float(size[1]), float(size[2])


def geometric_filter_and_resplit(
    clusters: list[Cluster],
    results: list[ClassifyResult],
    classifier: ClusterClassifier,
) -> tuple[list[Cluster], list[ClassifyResult]]:
    """
    Post-process classified clusters:

    1. If a cluster's bounding box violates the geometric bounds for its
       predicted class, demote it to 'background'.
    2. If a 'pedestrian' cluster is oversized (diagonal > bound), try to
       re-split it with a tighter eps and re-classify the fragments.
    """
    out_clusters: list[Cluster] = []
    out_results: list[ClassifyResult] = []

    resplit_queue: list[Cluster] = []

    for clust, res in zip(clusters, results):
        cls = res.class_name
        if cls == BACKGROUND_CLASS:
            out_clusters.append(clust)
            out_results.append(res)
            continue

        bounds = GEOMETRY_BOUNDS.get(cls)
        if bounds is None:
            out_clusters.append(clust)
            out_results.append(res)
            continue

        w, l, h = _bbox_dims(clust)
        diag = clust.bbox_diagonal

        plausible = (
            diag <= bounds["max_diag"]
            and w <= bounds["max_width"]
            and l <= bounds["max_length"]
            and h <= bounds["max_height"]
        )

        if plausible:
            out_clusters.append(clust)
            out_results.append(res)
        elif cls == "pedestrian" and diag > GEOMETRY_BOUNDS["pedestrian"]["max_diag"]:
            resplit_queue.append(clust)
        else:
            out_clusters.append(clust)
            out_results.append(ClassifyResult(
                class_name="background",
                class_idx=0,
                confidence=res.confidence,
            ))

    # --- Re-split oversized pedestrian clusters ------------------------------
    if resplit_queue:
        for big_clust in resplit_queue:
            sub_labels = cluster_objects(
                big_clust.points_xyz,
                eps=RESPLIT_EPS,
                min_samples=RESPLIT_MIN_SAMPLES,
            )
            sub_clusters = extract_clusters(
                big_clust.points_xyz,
                big_clust.points_full,
                sub_labels,
                min_points=MIN_CLUSTER_POINTS,
                max_points=MAX_CLUSTER_POINTS,
                max_diagonal=MAX_BBOX_DIAGONAL,
            )
            if sub_clusters:
                sub_results = classifier.classify_batch(sub_clusters)
                out_clusters.extend(sub_clusters)
                out_results.extend(sub_results)
            else:
                out_clusters.append(big_clust)
                out_results.append(ClassifyResult(
                    class_name="background",
                    class_idx=0,
                    confidence=0.0,
                ))

    return out_clusters, out_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    device = get_device()
    print(f"Device: {device}")

    seg_dir  = RESULTS_DIR / "segmentation"
    trk_dir  = RESULTS_DIR / "tracking"
    vis_dir  = RESULTS_DIR / "visualization"
    for d in (seg_dir, trk_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(DATA_DIR.glob("*.bin"))
    if not frame_paths:
        raise FileNotFoundError(f"No .bin files found in {DATA_DIR}")
    print(f"Found {len(frame_paths)} frame(s) in {DATA_DIR}")

    print(f"Loading classifier from {CHECKPOINT_PATH} …")
    classifier = ClusterClassifier(str(CHECKPOINT_PATH), device)

    MultiClassTracker.reset_ids()
    tracker = MultiClassTracker(
        max_age=TRACKER_MAX_AGE,
        min_hits=TRACKER_MIN_HITS,
    )

    all_tracked_per_frame: list = []
    per_frame_class_counts: list[dict[str, int]] = []
    all_tracking_json: list[dict] = []
    trail_history: dict[int, list[tuple[float, float]]] = {}

    # Pre-build the cKDTree import to avoid repeated import inside the loop
    from scipy.spatial import cKDTree  # noqa: F811

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Frames")):
        t_frame = time.time()

        # 1. Load
        points = load_frame(frame_path)
        xyz    = points[:, :3]
        N      = len(points)

        # 2. Ground removal (tighter threshold)
        ground_mask = segment_ground(
            xyz,
            distance_threshold=GROUND_DIST_THRESHOLD,
            max_ground_z=GROUND_MAX_Z,
        )
        obj_mask  = ~ground_mask
        xyz_obj   = xyz[obj_mask]
        pts_obj   = points[obj_mask]

        # 3. Two-pass DBSCAN clustering
        all_clusters = two_pass_cluster(xyz_obj, pts_obj)

        # 4. Classify clusters
        raw_results = classifier.classify_batch(all_clusters)

        # 5. Geometric post-filtering & re-splitting
        all_clusters, all_results = geometric_filter_and_resplit(
            all_clusters, raw_results, classifier,
        )

        # 6. Build per-point label array
        labels = np.zeros(N, dtype=np.int32)
        labels[obj_mask] = LABEL_BACKGROUND

        orig_indices = np.where(obj_mask)[0]
        tree = cKDTree(xyz_obj) if len(xyz_obj) > 0 else None

        for clust, res in zip(all_clusters, all_results):
            lbl = CLASS_TO_LABEL.get(res.class_name, LABEL_BACKGROUND)
            if tree is not None:
                _, nn_idx = tree.query(clust.points_xyz, k=1)
                labels[orig_indices[nn_idx]] = lbl

        # 7. Class-point counts
        label_names = {
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

        # 8. Prepare detections for tracker
        frame_detections: list[FrameDetection] = []
        for clust, res in zip(all_clusters, all_results):
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

        # 9. Update tracker
        tracked = tracker.update(frame_detections)
        all_tracked_per_frame.append(tracked)

        # 10. Save segmentation result
        np.savez_compressed(
            str(seg_dir / f"frame_{frame_idx:02d}.npz"),
            points=points,
            labels=labels,
        )

        # 11. Save tracking result
        frame_json = tracked_objects_to_json(tracked, frame_idx)
        all_tracking_json.extend(frame_json)
        with open(str(trk_dir / f"frame_{frame_idx:02d}.json"), "w") as fh:
            json.dump(frame_json, fh, indent=2)

        # 12. Update trail history
        for obj in tracked:
            tid = obj.track_id
            if tid not in trail_history:
                trail_history[tid] = []
            trail_history[tid].append((float(obj.centroid[0]), float(obj.centroid[1])))
            if len(trail_history[tid]) > TRAIL_LENGTH:
                trail_history[tid] = trail_history[tid][-TRAIL_LENGTH:]

        # 13. Save BEV visualization
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

        # 14. Per-frame console summary
        n_clusters   = len(all_clusters)
        cls_cnt      = Counter(r.class_name for r in all_results)
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

    with open(str(trk_dir / "all_frames.json"), "w") as fh:
        json.dump(all_tracking_json, fh, indent=2)

    save_trajectory_plot(
        all_tracked_per_frame,
        output_path=str(vis_dir / "trajectories.png"),
        xlim=BEV_XLIM,
        ylim=BEV_YLIM,
    )

    save_class_distribution_plot(
        per_frame_class_counts,
        output_path=str(vis_dir / "class_distribution.png"),
    )

    save_dashboard(
        all_tracked_per_frame,
        per_frame_class_counts,
        output_path=str(vis_dir / "dashboard.png"),
    )

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
    print(f"Improved pipeline complete in {(time.time() - t_total):.1f} s")
    print(f"Frames processed   : {len(frame_paths)}")
    print(f"Unique track IDs   : {len(total_tracks)}")
    for cls_name in ("car", "pedestrian", "bicyclist"):
        print(f"  {cls_name:<12}: {track_by_class.get(cls_name, 0)}")
    print("\nKey improvements over run_pipeline.py:")
    print(f"  Ground threshold : {GROUND_DIST_THRESHOLD} m (was 0.25 m)")
    print(f"  Fine DBSCAN eps  : {FINE_EPS} (pedestrian/cyclist pass)")
    print(f"  Coarse DBSCAN eps: {COARSE_EPS} (car/large-object pass)")
    print(f"  Min cluster pts  : {MIN_CLUSTER_POINTS} (was 10)")
    print(f"  Geometric filter : enabled (re-splits oversized ped clusters)")
    print(f"\nResults written to:")
    print(f"  Segmentation  → {seg_dir}")
    print(f"  Tracking      → {trk_dir}")
    print(f"  Visualization → {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
