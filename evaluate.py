"""
Test-set evaluation script for the trained PointNet++ classifier.

Usage:
    python evaluate.py

All paths and settings are controlled via the GLOBAL VARIABLES section below.
Run this script after training. It loads the best model checkpoint and
evaluates on the held-out test set.

Outputs:
    results/logs/test_report_<TIMESTAMP>.log
    results/confusion/test_confusion.png
"""

import os
import datetime
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import make_dataloader
from models.pointnet2 import PointNet2Classifier

# ===========================================================================
# GLOBAL VARIABLES — edit these to match your training configuration
# ===========================================================================

# --- Paths ------------------------------------------------------------------
DATA_ROOT       = "data"
TEST_DIR        = os.path.join(DATA_ROOT, "test")
RESULTS_DIR     = "results"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# --- Data -------------------------------------------------------------------
NUM_POINTS  = 256     # must match the value used during training
NUM_CLASSES = 4
CLASSES     = ["background", "bicyclist", "car", "pedestrian"]

# --- DataLoader -------------------------------------------------------------
BATCH_SIZE  = 64
NUM_WORKERS = 0   # set to 0 if DataLoader hangs on macOS/MPS

# ===========================================================================
# End of global variables
# ===========================================================================


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Logger (stdout + file)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w", buffering=1)

    def log(self, msg: str = ""):
        print(msg)
        self._file.write(msg + "\n")

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def save_confusion_matrix(cm: np.ndarray, classes: list, path: str, title: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in [
        (axes[0], cm,      "d",    "Counts"),
        (axes[1], cm_norm, ".2f",  "Normalised (recall)"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar=True,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} — {subtitle}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: torch.nn.Module, loader, device: torch.device):
    """
    Collect predictions and labels for the entire loader.

    Returns:
        all_labels (np.ndarray), all_preds (np.ndarray)
    """
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for pts, labels in tqdm(loader, desc="  Evaluating", leave=False):
            pts    = pts.to(device)
            logits = model(pts)
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device    = get_device()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(RESULTS_DIR, "logs", f"test_report_{timestamp}.log")
    logger    = Logger(log_path)

    logger.log("=" * 70)
    logger.log("PointNet++ — Test-Set Evaluation")
    logger.log("=" * 70)
    logger.log(f"  Timestamp       : {timestamp}")
    logger.log(f"  Device          : {device}")
    logger.log(f"  CHECKPOINT_PATH : {CHECKPOINT_PATH}")
    logger.log(f"  TEST_DIR        : {TEST_DIR}")
    logger.log(f"  NUM_POINTS      : {NUM_POINTS}")
    logger.log(f"  BATCH_SIZE      : {BATCH_SIZE}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    if not os.path.isfile(CHECKPOINT_PATH):
        logger.log(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")
        logger.log("Run train.py first.")
        logger.close()
        return

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    trained_classes = ckpt.get("classes", CLASSES)
    trained_epoch   = ckpt.get("epoch", "?")
    trained_f1      = ckpt.get("val_macro_f1", float("nan"))

    logger.log(f"  Checkpoint from epoch : {trained_epoch}")
    logger.log(f"  Val Macro-F1 at save  : {trained_f1:.4f}")
    logger.log(f"  Classes               : {trained_classes}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = PointNet2Classifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])

    # -----------------------------------------------------------------------
    # Test DataLoader
    # -----------------------------------------------------------------------
    logger.log("Loading test dataset ...")
    test_loader, test_dataset = make_dataloader(
        root_dir=TEST_DIR, num_points=NUM_POINTS, batch_size=BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS, shuffle=False,
        classes=CLASSES,
    )

    test_counts = test_dataset.class_counts()
    logger.log("  Test class distribution:")
    for cls, cnt in test_counts.items():
        logger.log(f"    {cls:<15}: {cnt:>6}")
    logger.log(f"  Total test samples: {len(test_dataset)}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    logger.log("Running inference ...")
    all_labels, all_preds = run_inference(model, test_loader, device)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    oa       = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wgtd_f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # -----------------------------------------------------------------------
    # Print & log
    # -----------------------------------------------------------------------
    logger.log("=" * 70)
    logger.log("Test Results")
    logger.log("=" * 70)
    logger.log(f"  Overall Accuracy  : {oa:.4f}  ({oa*100:.2f}%)")
    logger.log(f"  Macro F1-Score    : {macro_f1:.4f}   ← primary metric")
    logger.log(f"  Weighted F1-Score : {wgtd_f1:.4f}")
    logger.log("")
    logger.log("  Per-Class Classification Report:")
    for line in report.splitlines():
        logger.log("    " + line)
    logger.log("")

    logger.log("  Confusion Matrix (counts):")
    header = "    " + " ".join(f"{c[:6]:>8}" for c in CLASSES)
    logger.log(header)
    for i, row in enumerate(cm):
        row_str = "    " + f"{CLASSES[i][:6]:<8}" + " ".join(f"{v:>8}" for v in row)
        logger.log(row_str)
    logger.log("")

    # Per-class recall from CM (highlights safety-critical classes)
    logger.log("  Per-Class Recall (from confusion matrix):")
    for i, cls in enumerate(CLASSES):
        recall = cm[i, i] / max(cm[i].sum(), 1)
        logger.log(f"    {cls:<15}: {recall:.4f}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------------------------
    cm_path = os.path.join(RESULTS_DIR, "confusion", "test_confusion.png")
    save_confusion_matrix(cm, CLASSES, path=cm_path, title="Test Confusion Matrix")
    logger.log(f"  Confusion matrix saved → {cm_path}")
    logger.log(f"  Test report saved      → {log_path}")
    logger.log("")
    logger.log("Done.")
    logger.close()


if __name__ == "__main__":
    main()
