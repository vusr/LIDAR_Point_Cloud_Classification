"""
Cluster classifier that wraps the trained PointNet++ model.

The model was trained on pre-segmented LiDAR clusters from the same sensor
and scene domain as the optional-challenge data, making it directly reusable.
Each cluster is pre-processed the same way the training Dataset did:
    1. Resample / pad to NUM_POINTS (256)
    2. Centre at origin and scale to unit sphere
    3. Keep xyz only (model was trained on xyz)

The model outputs 4 logits: background / bicyclist / car / pedestrian.
Clusters predicted as background are filtered out by the caller (run_pipeline).

Usage:
    from classifier import ClusterClassifier
    clf = ClusterClassifier("../checkpoints/best_model.pth", device)
    results = clf.classify_batch(clusters)  # list of (class_name, confidence)
"""

import sys
import os
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the parent package importable regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.pointnet2 import PointNet2Classifier  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — must match the training configuration in train.py
# ---------------------------------------------------------------------------
NUM_POINTS: int = 256
CLASSES: list[str] = ["background", "bicyclist", "car", "pedestrian"]

# Batch size for inference — keep low to respect MPS memory
INFER_BATCH_SIZE: int = 64


class ClassifyResult(NamedTuple):
    class_name: str
    class_idx: int
    confidence: float   # softmax probability of the top class


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample / pad a point cloud to exactly n points."""
    cur = len(pts)
    if cur == n:
        return pts
    replace = cur < n
    idx = np.random.choice(cur, n, replace=replace)
    return pts[idx]


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Centre at origin; scale so the farthest point is on the unit sphere."""
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def _preprocess(points_xyz: np.ndarray) -> np.ndarray:
    """Full pre-processing chain: resample → normalise → float32."""
    pts = _resample(points_xyz[:, :3].astype(np.float32), NUM_POINTS)
    pts = _normalize(pts)
    return pts


class ClusterClassifier:
    """
    Wraps the trained PointNet++ checkpoint for inference on individual
    point-cloud clusters.

    Args:
        checkpoint_path: Path to `best_model.pth`.
        device:          torch.device — should be `mps` on M2 Mac.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.classes = CLASSES

        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)

        self.model = PointNet2Classifier(num_classes=len(CLASSES)).to(device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def classify_batch(self, clusters) -> list[ClassifyResult]:
        """
        Classify a list of Cluster objects.

        Args:
            clusters: list of clustering.Cluster instances.

        Returns:
            List of ClassifyResult (one per cluster, same order).
        """
        if not clusters:
            return []

        # Pre-process all clusters into a single numpy array
        preprocessed = np.stack(
            [_preprocess(c.points_xyz) for c in clusters], axis=0
        )  # (M, NUM_POINTS, 3)

        results: list[ClassifyResult] = []

        # Inference in mini-batches to keep MPS memory manageable
        for start in range(0, len(clusters), INFER_BATCH_SIZE):
            batch_np = preprocessed[start : start + INFER_BATCH_SIZE]
            batch_t  = torch.from_numpy(batch_np).to(self.device)  # (B, N, 3)

            logits = self.model(batch_t)                            # (B, 4)
            probs  = F.softmax(logits, dim=1).cpu().numpy()         # (B, 4)

            for p in probs:
                idx  = int(np.argmax(p))
                results.append(
                    ClassifyResult(
                        class_name=CLASSES[idx],
                        class_idx=idx,
                        confidence=float(p[idx]),
                    )
                )

        return results
