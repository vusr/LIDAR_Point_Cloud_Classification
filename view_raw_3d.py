"""
Interactive 3-D viewer for raw LiDAR frames (data/optional_challenge_data/).

No segmentation or tracking overlay — pure point cloud with three colour
modes that each reveal a different aspect of the scene:

  HEIGHT     Z-elevation mapped to the viridis colourmap.
             Low = purple/blue,  high = yellow/white.
             Good for spotting the ground plane, kerbs, and tall objects.

  INTENSITY  Laser return intensity (0–255) mapped to the plasma colourmap.
             Dark = low reflectivity (asphalt, dark clothing),
             bright = high reflectivity (road markings, signs, bike reflectors).

  RING       LiDAR beam channel (0–127) mapped to a rainbow palette.
             Each of the 128 horizontal scan rings gets a unique colour,
             making the scan pattern and angular resolution visible.

Controls
--------
  N  /  →   next frame
  B  /  ←   previous frame
  C          cycle colour mode  (Height → Intensity → Ring → …)
  R          reset camera to default position
  Q  / Esc   quit

Mouse:  left-drag = rotate   right-drag = zoom   middle-drag = pan

Usage
-----
  # from repo root (default data directory):
  python view_raw_3d.py

  # point at any folder of 5-channel float32 .bin files:
  python view_raw_3d.py --data_dir path/to/frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import vispy
from vispy.color import get_colormap
from vispy.scene import SceneCanvas, visuals
from vispy.scene.cameras import TurntableCamera

# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------
_REPO_ROOT       = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "optional_challenge_data"

# ---------------------------------------------------------------------------
# Colour-mode constants
# ---------------------------------------------------------------------------
COLOR_MODES = ("HEIGHT", "INTENSITY", "RING")

# Z range used for height colormap (clipped for visual clarity; most of the
# scene sits between -3 m and +5 m above sensor height)
Z_LOW  = -3.0   # metres — mapped to bottom of colormap
Z_HIGH =  5.0   # metres — mapped to top of colormap

N_RINGS = 128   # number of LiDAR beams in the sensor

MAX_RENDER_PTS = 120_000


# ---------------------------------------------------------------------------
# Colour builders
# ---------------------------------------------------------------------------

def _apply_colormap(values_01: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map normalised (0–1) float values to RGBA via a VisPy colormap."""
    cmap   = get_colormap(cmap_name)
    return cmap.map(values_01.astype(np.float32))   # (N, 4) float32


def _build_ring_palette(n: int = N_RINGS) -> np.ndarray:
    """
    Generate a (n, 4) RGBA palette — one colour per ring — using HSV cycling.
    Adjacent rings get adjacent hues, which groups nearby beams visually.
    """
    hues = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    # Convert HSV (hue, sat=0.9, val=0.9) → RGB using numpy (no colorsys needed)
    # Formula from Wikipedia: HSV → RGB
    h6   = hues * 6.0
    i    = np.floor(h6).astype(np.int32) % 6
    f    = h6 - np.floor(h6)
    p    = np.full(n, 0.9 * (1.0 - 0.9), dtype=np.float32)   # 0.9*(1-s)*v = 0.09
    q    = 0.9 * (1.0 - 0.9 * f)
    t    = 0.9 * (1.0 - 0.9 * (1.0 - f))
    v    = np.full(n, 0.9, dtype=np.float32)

    rgb  = np.zeros((n, 3), dtype=np.float32)
    for sector, (r_src, g_src, b_src) in enumerate([
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q),
    ]):
        mask = i == sector
        rgb[mask, 0] = r_src[mask]
        rgb[mask, 1] = g_src[mask]
        rgb[mask, 2] = b_src[mask]

    alpha   = np.ones((n, 1), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1)   # (n, 4)


_RING_PALETTE = _build_ring_palette(N_RINGS)


def build_colors(
    pts: np.ndarray,   # (N, 5): x y z intensity ring
    mode: str,
) -> np.ndarray:
    """Return (N, 4) RGBA float32 colours according to the chosen mode."""
    if mode == "HEIGHT":
        z_norm = np.clip((pts[:, 2] - Z_LOW) / (Z_HIGH - Z_LOW), 0.0, 1.0)
        return _apply_colormap(z_norm, "viridis")

    if mode == "INTENSITY":
        i_norm = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
        return _apply_colormap(i_norm, "plasma")

    # RING
    ring_ids = np.clip(pts[:, 4].astype(np.int32), 0, N_RINGS - 1)
    return _RING_PALETTE[ring_ids]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_frames(data_dir: Path) -> list[Path]:
    frames = sorted(data_dir.glob("*.bin"))
    if not frames:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")
    return frames


def load_frame(path: Path) -> np.ndarray:
    """Return (N, 5) float32 array: x, y, z, intensity, ring."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class RawVisualizer3D:
    """
    VisPy interactive 3-D viewer for raw LiDAR frames.
    Frames are pre-loaded into RAM for snappy N/B navigation.
    """

    def __init__(self, data_dir: Path) -> None:
        frame_paths = discover_frames(data_dir)

        print(f"\nLoading {len(frame_paths)} frame(s) from {data_dir} …")
        self._frames: list[np.ndarray] = []
        for i, fp in enumerate(frame_paths):
            pts = load_frame(fp)
            self._frames.append(pts)
            print(
                f"  frame {i:02d}: {len(pts):,} pts  "
                f"z=[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]  "
                f"intensity=[{pts[:,3].min():.0f}, {pts[:,3].max():.0f}]"
            )

        self._idx        = 0
        self._mode_idx   = 0   # index into COLOR_MODES

        self._build_canvas()
        self._render()

    # ------------------------------------------------------------------
    # Canvas
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

        self._pts_vis = visuals.Markers(parent=self.view.scene, antialias=False)
        visuals.XYZAxis(parent=self.view.scene)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    def _make_title(self) -> str:
        pts  = self._frames[self._idx]
        mode = COLOR_MODES[self._mode_idx]
        return (
            f"Raw LiDAR — Frame {self._idx + 1}/{len(self._frames)}  |  "
            f"{len(pts):,} pts  |  "
            f"Colour: {mode}  |  "
            f"[N/B] frame  [C] colour mode  [R] reset  [Q] quit"
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        pts  = self._frames[self._idx]
        mode = COLOR_MODES[self._mode_idx]

        # Subsample if needed
        n = len(pts)
        if n > MAX_RENDER_PTS:
            idx = np.random.choice(n, MAX_RENDER_PTS, replace=False)
            pts_r = pts[idx]
        else:
            pts_r = pts

        colors = build_colors(pts_r, mode)

        self._pts_vis.set_data(
            pts_r[:, :3],
            face_color=colors,
            edge_color=colors,
            size=2.0,
            edge_width=0,
        )

        self.canvas.title = self._make_title()
        self.canvas.update()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("N", "Right"):
            self._idx = (self._idx + 1) % len(self._frames)
            self._render()
        elif key in ("B", "Left"):
            self._idx = (self._idx - 1) % len(self._frames)
            self._render()
        elif key == "C":
            self._mode_idx = (self._mode_idx + 1) % len(COLOR_MODES)
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
        print("\nColour modes  (press C to cycle):")
        print("  HEIGHT     viridis  — purple/blue=low  yellow/white=high")
        print("  INTENSITY  plasma   — dark=low reflectivity  bright=high")
        print("  RING       rainbow  — one colour per LiDAR beam (0–127)")
        print("\nControls:")
        print("  N / →   next frame")
        print("  B / ←   previous frame")
        print("  C       cycle colour mode")
        print("  R       reset camera")
        print("  Q/Esc   quit")
        print("\nMouse:  left-drag = rotate   right-drag = zoom   middle-drag = pan\n")
        self.canvas.app.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3-D viewer for raw LiDAR frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        default=str(DEFAULT_DATA_DIR),
        help=(
            "Directory containing .bin frames (5-channel float32: x,y,z,intensity,ring). "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[error] Data directory not found: {data_dir}")
        sys.exit(1)

    vis = RawVisualizer3D(data_dir)
    vis.run()


if __name__ == "__main__":
    main()
