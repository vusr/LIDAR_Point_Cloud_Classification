"""
One-time script to physically split data/train/ into train and validation sets.
Each class is split independently (stratified) to preserve class proportions.

Files are MOVED (not copied) from data/train/<class>/ to data/val/<class>/.

Run this script ONCE before training. Running it again will not re-split
since the original train files will already have been moved.
"""

import os
import random
import shutil

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
TRAIN_DIR   = "data/train"
VAL_DIR     = "data/val"
VAL_SPLIT   = 0.3          # fraction of train files to move to val
RANDOM_SEED = 42
CLASSES     = ["background", "bicyclist", "car", "pedestrian"]
# ---------------------------------------------------------------------------


def split_class(cls: str, train_dir: str, val_dir: str, val_split: float, rng: random.Random) -> dict:
    """
    Stratified split for a single class.

    Randomly selects `val_split` fraction of files from train_dir/cls
    and moves them to val_dir/cls.

    Returns a dict with counts: {cls: {"train": int, "val": int}}.
    """
    src_dir = os.path.join(train_dir, cls)
    dst_dir = os.path.join(val_dir, cls)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Train class directory not found: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".bin")])
    total = len(all_files)

    if total == 0:
        print(f"  [{cls}] No .bin files found — skipping.")
        return {cls: {"train": 0, "val": 0}}

    n_val = max(1, round(total * val_split))
    val_files = set(rng.sample(all_files, n_val))

    moved = 0
    for fname in all_files:
        if fname in val_files:
            shutil.move(
                os.path.join(src_dir, fname),
                os.path.join(dst_dir, fname),
            )
            moved += 1

    n_train = total - moved
    return {cls: {"train": n_train, "val": moved}}


def main():
    rng = random.Random(RANDOM_SEED)

    print("=" * 60)
    print("Stratified Train / Val Split")
    print(f"  VAL_SPLIT   = {VAL_SPLIT}")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  TRAIN_DIR   = {TRAIN_DIR}")
    print(f"  VAL_DIR     = {VAL_DIR}")
    print("=" * 60)

    summary = {}
    for cls in CLASSES:
        result = split_class(cls, TRAIN_DIR, VAL_DIR, VAL_SPLIT, rng)
        summary.update(result)

    print("\nSplit Summary:")
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*43}")
    total_train, total_val = 0, 0
    for cls in CLASSES:
        n_train = summary[cls]["train"]
        n_val   = summary[cls]["val"]
        total   = n_train + n_val
        total_train += n_train
        total_val   += n_val
        print(f"  {cls:<15} {n_train:>8} {n_val:>8} {total:>8}")
    print(f"  {'-'*43}")
    print(f"  {'TOTAL':<15} {total_train:>8} {total_val:>8} {total_train+total_val:>8}")
    print("\nDone. Files have been moved. Do NOT run this script again.")


if __name__ == "__main__":
    main()
