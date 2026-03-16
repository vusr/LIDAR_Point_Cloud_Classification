"""
Training script for PointNet++ point cloud classification.

Usage:
    python train.py

All hyperparameters and paths are controlled via the GLOBAL VARIABLES section
below. No command-line arguments are required.

Outputs (written to RESULTS_DIR and CHECKPOINT_DIR):
    checkpoints/best_model.pth          ← best val macro-F1 checkpoint
    checkpoints/last_model.pth          ← final epoch checkpoint
    results/logs/training_<TS>.log      ← structured per-epoch log
    results/curves/loss_curve.png
    results/curves/accuracy_curve.png
    results/curves/macro_f1_curve.png
    results/confusion/val_best_confusion.png
"""

import os
import time
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on macOS/MPS
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import make_dataloader, CLASSES as DEFAULT_CLASSES
from models.pointnet2 import PointNet2Classifier

# ===========================================================================
# GLOBAL VARIABLES — edit these to tune the experiment
# ===========================================================================

# --- Paths ------------------------------------------------------------------
DATA_ROOT      = "data"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
VAL_DIR        = os.path.join(DATA_ROOT, "val")
RESULTS_DIR    = "results"
CHECKPOINT_DIR = "checkpoints"

# --- Data -------------------------------------------------------------------
NUM_POINTS         = 256      # every point cloud is resampled to this size
NUM_CLASSES        = 4
CLASSES            = ["background", "bicyclist", "car", "pedestrian"]
RANDOM_SEED        = 42
USE_AUGMENTATION   = True     # random rotation / jitter / scale on train set

# --- DataLoader -------------------------------------------------------------
BATCH_SIZE   = 32
NUM_WORKERS  = 0   # set to 0 if DataLoader hangs on macOS/MPS

# --- Optimiser --------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

# --- Scheduler (StepLR) -----------------------------------------------------
LR_STEP_SIZE = 20     # reduce LR every N epochs
LR_GAMMA     = 0.5    # multiplicative factor

# --- Training ---------------------------------------------------------------
NUM_EPOCHS         = 100
USE_CLASS_WEIGHTS  = True   # inverse-frequency weighting in CrossEntropyLoss
SAVE_BEST_ONLY     = True   # always saves last; this controls best checkpoint

# --- Model ------------------------------------------------------------------
DROPOUT_RATE  = 0.4
SA1_NPOINT    = 128
SA1_RADIUS    = 0.3
SA1_NSAMPLE   = 32
SA2_NPOINT    = 32
SA2_RADIUS    = 0.6
SA2_NSAMPLE   = 64

# --- Logging ----------------------------------------------------------------
LOG_EVERY_N_BATCHES = 20    # print batch-level progress every N batches

# ===========================================================================
# End of global variables
# ===========================================================================


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Logging helpers
# ---------------------------------------------------------------------------

class Logger:
    """Writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w", buffering=1)   # line-buffered

    def log(self, msg: str = ""):
        print(msg)
        self._file.write(msg + "\n")

    def close(self):
        self._file.close()


def format_epoch_summary(
    epoch: int,
    num_epochs: int,
    lr: float,
    tr_loss: float,
    tr_oa: float,
    tr_f1: float,
    va_loss: float,
    va_oa: float,
    va_f1: float,
    is_best: bool,
) -> str:
    marker = " *** BEST ***" if is_best else ""
    return (
        f"Epoch {epoch:03d}/{num_epochs:03d} | LR={lr:.2e} |"
        f" Loss: train={tr_loss:.4f}  val={va_loss:.4f} |"
        f" OA:   train={tr_oa:.4f}  val={va_oa:.4f} |"
        f" MacF1:train={tr_f1:.4f}  val={va_f1:.4f}"
        f"{marker}"
    )


def format_cls_report(report_str: str, prefix: str = "  ") -> str:
    """Indent a sklearn classification report for the log."""
    return "\n".join(prefix + line for line in report_str.splitlines())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_learning_curves(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    for key, title, fname in [
        ("loss",     "Loss",            "loss_curve.png"),
        ("oa",       "Overall Accuracy","accuracy_curve.png"),
        ("macro_f1", "Macro F1",        "macro_f1_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history[f"train_{key}"], label="Train", linewidth=1.8)
        ax.plot(epochs, history[f"val_{key}"],   label="Val",   linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, classes: list, path: str, title: str = "Confusion Matrix"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Normalised (row-normalised = recall per class)
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
# One epoch: train or evaluate
# ---------------------------------------------------------------------------

def run_epoch(
    model:       nn.Module,
    loader,
    criterion:   nn.Module,
    device:      torch.device,
    optimizer:   torch.optim.Optimizer = None,
    phase:       str = "train",
    logger:      Logger = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Run one full epoch.

    Returns:
        avg_loss, overall_accuracy, macro_f1, all_labels, all_preds
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (pts, labels) in enumerate(loader):
            pts    = pts.to(device)       # (B, N, 3)
            labels = labels.to(device)

            logits = model(pts)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if (
                logger is not None
                and is_train
                and (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0
            ):
                logger.log(
                    f"  [{phase}] batch {batch_idx+1:>5}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    avg_loss = total_loss / len(all_labels)
    oa       = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, oa, macro_f1, all_labels, all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(RESULTS_DIR, "logs", f"training_{timestamp}.log")
    logger    = Logger(log_path)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "curves"),    exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion"), exist_ok=True)

    logger.log("=" * 70)
    logger.log("PointNet++ Training — Point Cloud Classification")
    logger.log("=" * 70)
    logger.log(f"  Timestamp  : {timestamp}")
    logger.log(f"  Device     : {device}")
    logger.log(f"  NUM_POINTS : {NUM_POINTS}")
    logger.log(f"  BATCH_SIZE : {BATCH_SIZE}")
    logger.log(f"  NUM_EPOCHS : {NUM_EPOCHS}")
    logger.log(f"  LR         : {LEARNING_RATE}")
    logger.log(f"  WEIGHT_DECAY: {WEIGHT_DECAY}")
    logger.log(f"  LR_STEP    : {LR_STEP_SIZE}  GAMMA={LR_GAMMA}")
    logger.log(f"  AUGMENT    : {USE_AUGMENTATION}")
    logger.log(f"  CLASS_WGTS : {USE_CLASS_WEIGHTS}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    logger.log("Loading datasets ...")
    train_loader, train_dataset = make_dataloader(
        root_dir=TRAIN_DIR, num_points=NUM_POINTS, batch_size=BATCH_SIZE,
        augment=USE_AUGMENTATION, num_workers=NUM_WORKERS, shuffle=True,
        classes=CLASSES,
    )
    val_loader, _ = make_dataloader(
        root_dir=VAL_DIR, num_points=NUM_POINTS, batch_size=BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS, shuffle=False,
        classes=CLASSES,
    )

    class_counts = train_dataset.class_counts()
    logger.log("  Train class distribution:")
    for cls, cnt in class_counts.items():
        logger.log(f"    {cls:<15}: {cnt:>6}")
    logger.log(f"  Total train : {len(train_dataset):>6}")
    logger.log(f"  Total val   : {len(val_loader.dataset):>6}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    if USE_CLASS_WEIGHTS:
        class_weights = train_dataset.class_weights().to(device)
        logger.log("  Class weights (inverse-freq):")
        for cls, w in zip(CLASSES, class_weights.cpu()):
            logger.log(f"    {cls:<15}: {w:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = PointNet2Classifier(
        num_classes  = NUM_CLASSES,
        sa1_npoint   = SA1_NPOINT,
        sa1_radius   = SA1_RADIUS,
        sa1_nsample  = SA1_NSAMPLE,
        sa2_npoint   = SA2_NPOINT,
        sa2_radius   = SA2_RADIUS,
        sa2_nsample  = SA2_NSAMPLE,
        dropout_rate = DROPOUT_RATE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n  Model params: {n_params:,}")
    logger.log("")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "train_oa":   [], "val_oa":   [],
        "train_macro_f1": [], "val_macro_f1": [],
    }

    best_val_f1    = -1.0
    best_epoch     = 0
    best_cm        = None
    best_labels    = None
    best_preds     = None

    logger.log("=" * 70)
    logger.log("Training")
    logger.log("=" * 70)
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.get_last_lr()[0]

        # Train
        tr_loss, tr_oa, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, phase="train", logger=logger,
        )

        # Validate
        va_loss, va_oa, va_f1, va_labels, va_preds = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, phase="val", logger=None,
        )

        scheduler.step()

        # Record history
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_oa"].append(tr_oa)
        history["val_oa"].append(va_oa)
        history["train_macro_f1"].append(tr_f1)
        history["val_macro_f1"].append(va_f1)

        is_best = va_f1 > best_val_f1
        if is_best:
            best_val_f1  = va_f1
            best_epoch   = epoch
            best_labels  = va_labels
            best_preds   = va_preds
            best_cm      = confusion_matrix(va_labels, va_preds, labels=list(range(NUM_CLASSES)))
            if SAVE_BEST_ONLY:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "val_macro_f1": va_f1, "classes": CLASSES},
                    os.path.join(CHECKPOINT_DIR, "best_model.pth"),
                )

        # Per-epoch summary
        summary = format_epoch_summary(
            epoch, NUM_EPOCHS, current_lr,
            tr_loss, tr_oa, tr_f1,
            va_loss, va_oa, va_f1,
            is_best,
        )
        logger.log(summary)

        # Every 10 epochs print the val classification report
        if epoch % 10 == 0 or epoch == 1 or is_best:
            report = classification_report(
                va_labels, va_preds,
                target_names=CLASSES,
                digits=4,
                zero_division=0,
            )
            logger.log("")
            logger.log("  Val Classification Report:")
            logger.log(format_cls_report(report))
            logger.log("")

    # -----------------------------------------------------------------------
    # Save last checkpoint
    # -----------------------------------------------------------------------
    torch.save(
        {"epoch": NUM_EPOCHS, "model_state": model.state_dict(),
         "optimizer_state": optimizer.state_dict(),
         "val_macro_f1": va_f1, "classes": CLASSES},
        os.path.join(CHECKPOINT_DIR, "last_model.pth"),
    )

    elapsed = time.time() - t_start
    logger.log("=" * 70)
    logger.log(f"Training complete in {elapsed/60:.1f} min")
    logger.log(f"Best val Macro-F1 = {best_val_f1:.4f} at epoch {best_epoch}")
    logger.log("=" * 70)

    # -----------------------------------------------------------------------
    # Final train-set metrics (with best checkpoint)
    # -----------------------------------------------------------------------
    logger.log("\nRunning final train-set evaluation with best model ...")
    best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    _, _, _, tr_labels_final, tr_preds_final = run_epoch(
        model, train_loader, criterion, device, phase="val", logger=None,
    )
    tr_report = classification_report(
        tr_labels_final, tr_preds_final,
        target_names=CLASSES, digits=4, zero_division=0,
    )
    logger.log("\n  Final Train Classification Report (best model):")
    logger.log(format_cls_report(tr_report))

    # -----------------------------------------------------------------------
    # Best-val confusion matrix
    # -----------------------------------------------------------------------
    save_confusion_matrix(
        best_cm,
        CLASSES,
        path=os.path.join(RESULTS_DIR, "confusion", "val_best_confusion.png"),
        title=f"Validation Confusion Matrix (Epoch {best_epoch})",
    )
    logger.log(f"\n  Confusion matrix saved → results/confusion/val_best_confusion.png")

    # -----------------------------------------------------------------------
    # Learning curves
    # -----------------------------------------------------------------------
    save_learning_curves(history, out_dir=os.path.join(RESULTS_DIR, "curves"))
    logger.log("  Learning curves saved  → results/curves/")

    logger.log("\nDone.")
    logger.close()


if __name__ == "__main__":
    main()
