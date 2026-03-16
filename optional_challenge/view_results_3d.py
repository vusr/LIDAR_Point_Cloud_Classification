"""
Interactive 3-D viewer for optional-challenge segmentation + tracking results.

Each segmented frame (.npz) is displayed as a coloured point cloud in a VisPy
3-D window.  Tracked objects are overlaid as wireframe bounding boxes with
track-ID labels positioned above each box.

Controls
--------
  N  /  →   next frame
  B  /  ←   previous frame
  G          toggle ground points on / off  (hidden by default — they clutter)
  A          toggle background / noise on / off
  R          reset camera to default position
  Q / Esc    quit

Mouse: left-drag = rotate   right-drag = zoom   middle-drag = pan

Colour legend
-------------
  ground      dark grey   (label 0) — hidden by default, press G to show
  background  light grey  (label 1)
  car         RED         (label 2)   bboxes: bright red wireframe
  pedestrian  BLUE        (label 3)   bboxes: bright blue wireframe
  bicyclist   GREEN       (label 4)   bboxes: bright green wireframe

Usage
-----
  # from repo root:
  python optional_challenge/view_results_3d.py

  # view the SalsaNext pipeline results instead:
  python optional_challenge/view_results_3d.py \\
      --results_dir optional_challenge/results_salsanext

  # point at any directory that has segmentation/ and tracking/ sub-folders:
  python optional_challenge/view_results_3d.py --results_dir /path/to/results
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import vispy
from vispy.scene import SceneCanvas, visuals
from vispy.scene.cameras import TurntableCamera

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_THIS_DIR           = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = _THIS_DIR / "results"

# ---------------------------------------------------------------------------
# Semantic class definitions
# ---------------------------------------------------------------------------
LABEL_TO_NAME: dict[int, str] = {
    0: "ground",
    1: "background",
    2: "car",
    3: "pedestrian",
    4: "bicyclist",
}

# RGBA (0–1) used when drawing each point class.
# Ground and background are kept very transparent so objects stand out.
CLASS_RGBA: dict[str, tuple[float, float, float, float]] = {
    "ground":     (0.25, 0.25, 0.30, 0.12),
    "background": (0.55, 0.55, 0.55, 0.28),
    "car":        (0.95, 0.22, 0.18, 0.95),
    "pedestrian": (0.15, 0.52, 0.95, 0.95),
    "bicyclist":  (0.12, 0.85, 0.38, 0.95),
}

# RGBA used for wireframe bounding-box edges and text labels.
BBOX_RGBA: dict[str, tuple[float, float, float, float]] = {
    "car":        (1.00, 0.45, 0.40, 1.0),
    "pedestrian": (0.38, 0.74, 1.00, 1.0),
    "bicyclist":  (0.30, 1.00, 0.52, 1.0),
}

# Maximum number of points rendered per frame.
# Object-class points are always kept; ground / background are subsampled.
MAX_RENDER_PTS = 100_000


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _discover_frames(
    seg_dir: Path, trk_dir: Path
) -> list[tuple[Path, Path, int]]:
    """Return sorted list of (seg_path, trk_path, frame_idx) triples."""
    seg_files = sorted(seg_dir.glob("frame_*.npz"))
    if not seg_files:
        raise FileNotFoundError(f"No frame_XX.npz files found in {seg_dir}")
    frames = []
    for sp in seg_files:
        idx = int(sp.stem.split("_")[1])
        tp  = trk_dir / f"frame_{idx:02d}.json"
        frames.append((sp, tp, idx))
    return frames


def _load_frame(
    seg_path: Path, trk_path: Path
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load one segmented frame: (N,3) points, (N,) int labels, list of tracks."""
    data   = np.load(str(seg_path))
    pts    = data["points"][:, :3].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    tracks: list[dict] = []
    if trk_path.exists():
        with open(trk_path) as fh:
            tracks = json.load(fh)
    return pts, labels, tracks


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _build_point_colors(
    labels: np.ndarray,
    show_ground: bool,
    show_bg: bool,
) -> np.ndarray:
    """Return (N, 4) float32 RGBA array; set alpha=0 for hidden classes."""
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    for lbl, name in LABEL_TO_NAME.items():
        mask = labels == lbl
        if not mask.any():
            continue
        rgba = list(CLASS_RGBA[name])
        if lbl == 0 and not show_ground:
            rgba[3] = 0.0
        if lbl == 1 and not show_bg:
            rgba[3] = 0.0
        colors[mask] = rgba
    return colors


def _subsample_indices(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Always keep ALL object points (car/ped/bike).
    Subsample ground+background so the total stays under MAX_RENDER_PTS.
    """
    obj_idx = np.where(labels >= 2)[0]
    bg_idx  = np.where(labels < 2)[0]
    budget  = max(0, MAX_RENDER_PTS - len(obj_idx))
    if len(bg_idx) > budget:
        bg_idx = np.random.choice(bg_idx, budget, replace=False)
    return np.concatenate([obj_idx, bg_idx])


def _build_bbox_geometry(
    tracks: list[dict],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Build (M*24, 3) positions and (M*24, 4) RGBA colours for all track
    bounding boxes, using connect='segments' mode (each consecutive pair of
    vertices is one edge).  Returns (None, None) when there are no tracks.
    """
    if not tracks:
        return None, None

    segs_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    for obj in tracks:
        x0, y0, z0 = (float(v) for v in obj["bbox_min"])
        x1, y1, z1 = (float(v) for v in obj["bbox_max"])

        # 12 edges × 2 endpoints = 24 rows
        edges = np.array([
            # bottom face
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            # top face
            [x0, y0, z1], [x1, y0, z1],
            [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1],
            [x0, y1, z1], [x0, y0, z1],
            # vertical pillars
            [x0, y0, z0], [x0, y0, z1],
            [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1],
            [x0, y1, z0], [x0, y1, z1],
        ], dtype=np.float32)

        color = BBOX_RGBA.get(obj["class"], (1.0, 1.0, 1.0, 1.0))
        segs_list.append(edges)
        cols_list.append(np.tile(color, (24, 1)).astype(np.float32))

    return np.vstack(segs_list), np.vstack(cols_list)


def _build_label_data(
    tracks: list[dict],
) -> tuple[list[str], np.ndarray]:
    """
    Returns (texts, positions) for track labels.
    Labels are positioned 0.5 m above the top face of each bounding box.
    """
    if not tracks:
        return [], np.zeros((0, 3), dtype=np.float32)

    texts: list[str] = []
    positions: list[list[float]] = []

    for obj in tracks:
        cx    = float(obj["centroid"][0])
        cy    = float(obj["centroid"][1])
        z_top = float(obj["bbox_max"][2]) + 0.5
        cls   = obj["class"]
        tid   = obj["track_id"]
        conf  = obj["confidence"]
        texts.append(f"#{tid} {cls[:3].upper()} {conf:.0%}")
        positions.append([cx, cy, z_top])

    return texts, np.array(positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------

class Visualizer3D:
    """
    VisPy-based interactive 3-D viewer.

    Pre-loads all frames into memory at startup for instant N/B navigation.
    """

    def __init__(self, results_dir: Path) -> None:
        seg_dir = results_dir / "segmentation"
        trk_dir = results_dir / "tracking"

        frame_list = _discover_frames(seg_dir, trk_dir)
        print(f"\nLoading {len(frame_list)} frame(s) from {results_dir} …")

        self._data: list[tuple[np.ndarray, np.ndarray, list[dict]]] = []
        for seg_path, trk_path, fidx in frame_list:
            pts, lbls, tracks = _load_frame(seg_path, trk_path)
            self._data.append((pts, lbls, tracks))

            cnt   = Counter(LABEL_TO_NAME.get(int(l), "?") for l in lbls)
            t_cnt = Counter(t["class"] for t in tracks)
            print(
                f"  frame {fidx:02d}: {len(pts):,} pts — "
                f"ground={cnt['ground']:,}  bg={cnt['background']:,}  "
                f"car={cnt['car']:,}  ped={cnt['pedestrian']:,}  "
                f"bike={cnt['bicyclist']:,}  |  "
                f"tracks: car={t_cnt.get('car', 0)}  "
                f"ped={t_cnt.get('pedestrian', 0)}  "
                f"bike={t_cnt.get('bicyclist', 0)}"
            )

        self._idx         = 0
        self._show_ground = False   # ground clutters the view; hidden by default
        self._show_bg     = True

        self._build_canvas()
        self._render()

    # ------------------------------------------------------------------
    # Canvas setup
    # ------------------------------------------------------------------

    def _build_canvas(self) -> None:
        self.canvas = SceneCanvas(
            title=self._make_title(),
            keys="interactive",
            show=True,
            size=(1440, 900),
            bgcolor="#0d0d1a",
        )
        self.canvas.events.key_press.connect(self._on_key)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = TurntableCamera(
            elevation=25,
            azimuth=225,
            distance=90,
            center=(0.0, 0.0, 0.0),
            fov=45,
        )

        # Visuals created once and updated in-place each frame
        self._pts_vis = visuals.Markers(parent=self.view.scene, antialias=False)
        self._box_vis = visuals.Line(
            parent=self.view.scene, connect="segments", antialias=True,
        )
        self._lbl_vis = visuals.Text(
            parent=self.view.scene,
            bold=True,
            font_size=8,
            anchor_x="center",
            anchor_y="bottom",
            color="white",
        )
        visuals.XYZAxis(parent=self.view.scene)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    def _make_title(self) -> str:
        pts, lbls, tracks = self._data[self._idx]
        cnt   = Counter(LABEL_TO_NAME.get(int(l), "?") for l in lbls)
        t_cnt = Counter(t["class"] for t in tracks)
        g_str = "ON" if self._show_ground else "off"
        a_str = "ON" if self._show_bg     else "off"
        return (
            f"Frame {self._idx + 1}/{len(self._data)}  |  "
            f"pts: car={cnt['car']:,}  ped={cnt['pedestrian']:,}  "
            f"bike={cnt['bicyclist']:,}  bg={cnt['background']:,}  "
            f"ground={cnt['ground']:,}  |  "
            f"tracks → car:{t_cnt.get('car', 0)}  ped:{t_cnt.get('pedestrian', 0)}  "
            f"bike:{t_cnt.get('bicyclist', 0)}  |  "
            f"[N/B] frame  [G] ground={g_str}  [A] bg={a_str}  [R] reset  [Q] quit"
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        pts, lbls, tracks = self._data[self._idx]

        # ---- Point cloud ---------------------------------------------------
        colors   = _build_point_colors(lbls, self._show_ground, self._show_bg)
        keep_idx = _subsample_indices(lbls)

        self._pts_vis.set_data(
            pts[keep_idx],
            face_color=colors[keep_idx],
            edge_color=colors[keep_idx],
            size=2.5,
            edge_width=0,
        )

        # ---- Bounding boxes ------------------------------------------------
        seg_pos, seg_col = _build_bbox_geometry(tracks)
        if seg_pos is not None:
            self._box_vis.set_data(pos=seg_pos, color=seg_col, width=2.5)
        else:
            # Invisible placeholder so the visual is not in an uninitialised state
            self._box_vis.set_data(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=np.zeros((2, 4), dtype=np.float32),
                width=1.0,
            )

        # ---- Track-ID labels -----------------------------------------------
        texts, lbl_pos = _build_label_data(tracks)
        if texts:
            self._lbl_vis.text = texts
            self._lbl_vis.pos  = lbl_pos
        else:
            # VisPy Text needs at least a non-empty string to avoid internal errors
            self._lbl_vis.text = " "

        self.canvas.title = self._make_title()
        self.canvas.update()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("N", "Right"):
            self._idx = (self._idx + 1) % len(self._data)
            self._render()
        elif key in ("B", "Left"):
            self._idx = (self._idx - 1) % len(self._data)
            self._render()
        elif key == "G":
            self._show_ground = not self._show_ground
            self._render()
        elif key == "A":
            self._show_bg = not self._show_bg
            self._render()
        elif key == "R":
            self.view.camera.set_range()
            self.canvas.update()
        elif key in ("Q", "Escape"):
            self.canvas.close()
            try:
                self.canvas.app.quit()
            except Exception:
                pass

    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\nColour legend:")
        print("  ground      dark grey   [hidden — press G to show]")
        print("  background  light grey")
        print("  car         RED         (bbox: bright red wireframe)")
        print("  pedestrian  BLUE        (bbox: bright blue wireframe)")
        print("  bicyclist   GREEN       (bbox: bright green wireframe)")
        print("\nControls:")
        print("  N / →   next frame")
        print("  B / ←   previous frame")
        print("  G       toggle ground points")
        print("  A       toggle background points")
        print("  R       reset camera")
        print("  Q/Esc   quit")
        print("\nMouse:  left-drag = rotate   right-drag = zoom   middle-drag = pan\n")
        self.canvas.app.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3-D viewer for segmentation + tracking results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Directory containing segmentation/ and tracking/ sub-folders "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}")
        print("  Run the pipeline first:  cd optional_challenge && python run_pipeline.py")
        sys.exit(1)

    vis = Visualizer3D(results_dir)
    vis.run()


if __name__ == "__main__":
    main()
