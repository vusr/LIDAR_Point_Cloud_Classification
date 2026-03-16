"""
PyTorch Dataset and DataLoader utilities for point cloud classification.

Directory structure expected:
    <split_dir>/
        background/   *.bin
        bicyclist/    *.bin
        car/          *.bin
        pedestrian/   *.bin

Each .bin file stores float32 x, y, z values (3 floats per point).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants (must match across dataset.py, train.py, evaluate.py)
# ---------------------------------------------------------------------------
CLASSES     = ["background", "bicyclist", "car", "pedestrian"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
# ---------------------------------------------------------------------------


class PointCloudDataset(Dataset):
    """
    Loads pre-segmented LiDAR point cloud clusters for classification.

    Each sample is normalised and resampled/padded to a fixed number of
    points so that batches can be stacked.

    Args:
        root_dir:      path to a split directory (train / val / test).
        num_points:    fixed number of points every sample is resampled to.
        augment:       if True, apply random rotation, jitter and scale.
        classes:       ordered list of class names (subdirectory names).
    """

    def __init__(
        self,
        root_dir: str,
        num_points: int = 256,
        augment: bool = False,
        classes: list = None,
    ):
        if classes is None:
            classes = CLASSES

        self.root_dir   = root_dir
        self.num_points = num_points
        self.augment    = augment
        self.classes    = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: list[tuple[str, int]] = []   # (file_path, label_idx)
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".bin"):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .bin files found in {root_dir}. "
                               "Check directory structure and class names.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> dict:
        """Returns {class_name: count} for the dataset."""
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss, one per class.
        Weight_i = total / (num_classes * count_i).
        """
        counts = self.class_counts()
        total  = sum(counts.values())
        n      = len(self.classes)
        weights = torch.tensor(
            [total / (n * max(counts[c], 1)) for c in self.classes],
            dtype=torch.float32,
        )
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_points(self, path: str) -> np.ndarray:
        """Load x, y, z from a binary file → (N, 3) float32 array."""
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
        return pts

    def _resample(self, pts: np.ndarray) -> np.ndarray:
        """
        Resample or pad point cloud to exactly self.num_points points.

        - n > num_points : random sample without replacement
        - n < num_points : random sample WITH replacement (repeat points)
        - n == num_points: unchanged
        """
        n = len(pts)
        if n == self.num_points:
            return pts
        replace = n < self.num_points
        idx = np.random.choice(n, self.num_points, replace=replace)
        return pts[idx]

    @staticmethod
    def _normalize(pts: np.ndarray) -> np.ndarray:
        """
        Centre the cluster at the origin and scale so that the furthest
        point from the centroid lands on a unit sphere.
        """
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
        return pts

    @staticmethod
    def _augment(pts: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations suitable for LiDAR point clouds:
          - Random rotation around the z-axis (yaw)
          - Gaussian jitter on all coordinates
          - Random uniform scale
        """
        # Yaw rotation
        theta = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        Rz    = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]], dtype=np.float32)
        pts   = pts @ Rz.T

        # Gaussian jitter (clipped to avoid outliers)
        jitter = np.clip(np.random.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
        pts    = pts + jitter

        # Random scale ±10%
        scale = np.random.uniform(0.9, 1.1)
        pts   = (pts * scale).astype(np.float32)

        return pts

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        pts = self._load_points(path)
        pts = self._resample(pts)
        pts = self._normalize(pts)

        if self.augment:
            pts = self._augment(pts)

        # Shape: (num_points, 3) → tensor
        pts_tensor = torch.from_numpy(pts.astype(np.float32))
        return pts_tensor, label


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_dataloader(
    root_dir: str,
    num_points: int,
    batch_size: int,
    augment: bool,
    num_workers: int,
    shuffle: bool = True,
    classes: list = None,
) -> DataLoader:
    """Create a DataLoader for the given split directory."""
    dataset = PointCloudDataset(
        root_dir=root_dir,
        num_points=num_points,
        augment=augment,
        classes=classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,   # pin_memory not supported on MPS
        drop_last=False,
    )
    return loader, dataset
