"""
Inference script for the PointNet++ point cloud classifier.

Classifies one or more pre-segmented LiDAR cluster files (.bin) using
the trained model checkpoint.  Each .bin file should contain float32
(x, y, z) values, i.e. the same format used by the training and
evaluation pipelines.

Usage
-----
  # Classify a single file:
  python infer.py path/to/cluster.bin

  # Classify every .bin file in a directory:
  python infer.py path/to/clusters/

  # Use a custom checkpoint:
  python infer.py path/to/cluster.bin --checkpoint checkpoints/best_model.pth

  # Show class probabilities alongside the top prediction:
  python infer.py path/to/cluster.bin --probs

Outputs
-------
  Prints a table to stdout with one row per input file:
    <file>  <predicted_class>  [<confidence>]  [<all_probs>]

  Exit code 0 on success; non-zero on error (missing files, bad checkpoint).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.pointnet2 import PointNet2Classifier

# ---------------------------------------------------------------------------
# Defaults (must match training configuration)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"
DEFAULT_NUM_POINTS = 256
DEFAULT_CLASSES    = ["background", "bicyclist", "car", "pedestrian"]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Point cloud preprocessing (mirrors dataset.py)
# ---------------------------------------------------------------------------

def _resample(pts: np.ndarray, num_points: int) -> np.ndarray:
    n = len(pts)
    if n == num_points:
        return pts
    replace = n < num_points
    idx = np.random.choice(n, num_points, replace=replace)
    return pts[idx]


def _normalize(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def preprocess(path: Path, num_points: int) -> torch.Tensor:
    """
    Load a .bin cluster file and return a (1, num_points, 3) float32 tensor
    ready for the model, applying the same normalisation as during training.
    """
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 3)
    pts = _resample(pts, num_points)
    pts = _normalize(pts)
    return torch.from_numpy(pts.astype(np.float32)).unsqueeze(0)   # (1, N, 3)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PointNet2Classifier, list[str]]:
    """Load the PointNet++ model from a checkpoint file."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        print(f"[error] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("  Run train.py first, or point --checkpoint at a valid .pth file.",
              file=sys.stderr)
        sys.exit(1)

    ckpt    = torch.load(str(ckpt_path), map_location=device)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    epoch   = ckpt.get("epoch", "?")
    val_f1  = ckpt.get("val_macro_f1", float("nan"))

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Saved at epoch  : {epoch}")
    print(f"  Val Macro-F1    : {val_f1:.4f}")
    print(f"  Classes         : {classes}")
    print(f"  Device          : {device}\n")

    model = PointNet2Classifier(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


@torch.no_grad()
def infer_files(
    file_paths: list[Path],
    model: PointNet2Classifier,
    classes: list[str],
    device: torch.device,
    num_points: int,
    show_probs: bool,
) -> None:
    """Run inference on a list of .bin files and print results."""
    col_w = max(len(str(p)) for p in file_paths)

    header = f"{'File':<{col_w}}  {'Predicted':<12}  {'Confidence':>11}"
    if show_probs:
        header += "  " + "  ".join(f"{c:>10}" for c in classes)
    print(header)
    print("-" * len(header))

    for path in file_paths:
        if not path.is_file():
            print(f"{str(path):<{col_w}}  [error: file not found]")
            continue

        try:
            pts = preprocess(path, num_points).to(device)
        except Exception as exc:
            print(f"{str(path):<{col_w}}  [error: {exc}]")
            continue

        logits = model(pts)                              # (1, C)
        probs  = F.softmax(logits, dim=1)[0]             # (C,)
        pred_idx  = int(probs.argmax().item())
        pred_cls  = classes[pred_idx]
        confidence = float(probs[pred_idx].item())

        row = f"{str(path):<{col_w}}  {pred_cls:<12}  {confidence:>10.2%}"
        if show_probs:
            row += "  " + "  ".join(f"{float(p.item()):>10.4f}" for p in probs)
        print(row)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PointNet++ point cloud classifier — inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "One or more .bin files or directories containing .bin files. "
            "Each file must store float32 (x, y, z) point coordinates."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=f"Number of points to resample each cloud to (default: {DEFAULT_NUM_POINTS})",
    )
    parser.add_argument(
        "--probs",
        action="store_true",
        help="Also print per-class softmax probabilities",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for point resampling (default: 0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Expand inputs: files + directories
    file_paths: list[Path] = []
    for entry in args.input:
        p = Path(entry)
        if p.is_dir():
            found = sorted(p.glob("*.bin"))
            if not found:
                print(f"[warn] No .bin files found in directory: {p}", file=sys.stderr)
            file_paths.extend(found)
        elif p.suffix == ".bin":
            file_paths.append(p)
        else:
            print(f"[warn] Skipping non-.bin input: {p}", file=sys.stderr)

    if not file_paths:
        print("[error] No valid .bin input files found.", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    model, classes = load_model(args.checkpoint, device)
    infer_files(file_paths, model, classes, device, args.num_points, args.probs)


if __name__ == "__main__":
    main()
