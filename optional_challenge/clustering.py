"""
Instance clustering for non-ground LiDAR points using DBSCAN.

After ground removal, nearby non-ground points that belong to the same
physical object are grouped into clusters.  Each cluster is then described
by its constituent points plus a 3-D bounding box and centroid, which are
the inputs required by the downstream classifier and tracker.

Usage (standalone):
    from clustering import cluster_objects, extract_clusters
    labels = cluster_objects(points_xyz)
    clusters = extract_clusters(points_xyz, points_full, labels)
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Tuneable limits (applied AFTER DBSCAN to filter noise / implausible blobs)
# ---------------------------------------------------------------------------

# Minimum number of points a cluster must have to be kept
MIN_CLUSTER_POINTS: int = 10

# Maximum number of points (very large blobs are usually walls / merged objects)
MAX_CLUSTER_POINTS: int = 8000

# Maximum bounding-box diagonal in metres (filters huge merged clusters)
MAX_BBOX_DIAGONAL: float = 20.0


@dataclass
class Cluster:
    """All information about one extracted instance cluster."""

    cluster_id: int
    # xyz only (N, 3) — used for classification
    points_xyz: np.ndarray
    # full 5-channel points (N, 5) — stored in results
    points_full: np.ndarray
    # 3-D bounding box corners
    bbox_min: np.ndarray      # (3,) — min  x, y, z
    bbox_max: np.ndarray      # (3,) — max  x, y, z
    centroid: np.ndarray      # (3,) — mean x, y, z
    num_points: int = field(init=False)

    def __post_init__(self):
        self.num_points = len(self.points_xyz)

    @property
    def bbox_size(self) -> np.ndarray:
        """Width, length, height of the bounding box (3,)."""
        return self.bbox_max - self.bbox_min

    @property
    def bbox_diagonal(self) -> float:
        return float(np.linalg.norm(self.bbox_size))


def cluster_objects(
    points_xyz: np.ndarray,
    eps: float = 0.6,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Run DBSCAN on non-ground xyz points.

    Args:
        points_xyz:  (N, 3) xyz coordinates of non-ground points.
        eps:         Neighbourhood radius in metres.
        min_samples: Minimum points to form a core point.

    Returns:
        labels: (N,) int — cluster index per point; -1 = noise.
    """
    if len(points_xyz) == 0:
        return np.array([], dtype=np.int32)

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(points_xyz).astype(np.int32)
    return labels


def extract_clusters(
    points_xyz: np.ndarray,
    points_full: np.ndarray,
    labels: np.ndarray,
    min_points: int = MIN_CLUSTER_POINTS,
    max_points: int = MAX_CLUSTER_POINTS,
    max_diagonal: float = MAX_BBOX_DIAGONAL,
) -> list[Cluster]:
    """
    Convert DBSCAN label array into a list of Cluster objects, filtering
    out noise (-1), under-populated clusters, and oversized blobs.

    Args:
        points_xyz:   (N, 3) xyz of non-ground points.
        points_full:  (N, 5) full channel array of the same points.
        labels:       (N,) DBSCAN output labels.
        min_points:   Discard clusters with fewer points.
        max_points:   Discard clusters with more points.
        max_diagonal: Discard clusters whose bounding box diagonal exceeds this.

    Returns:
        List of Cluster objects, one per valid instance.
    """
    clusters: list[Cluster] = []

    unique_ids = np.unique(labels)
    for cid in unique_ids:
        if cid == -1:
            continue  # DBSCAN noise

        mask = labels == cid
        n = int(mask.sum())

        if n < min_points or n > max_points:
            continue

        pts_xyz  = points_xyz[mask]
        pts_full = points_full[mask]

        bbox_min = pts_xyz.min(axis=0)
        bbox_max = pts_xyz.max(axis=0)
        centroid = pts_xyz.mean(axis=0)

        diagonal = float(np.linalg.norm(bbox_max - bbox_min))
        if diagonal > max_diagonal:
            continue

        clusters.append(
            Cluster(
                cluster_id=int(cid),
                points_xyz=pts_xyz,
                points_full=pts_full,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                centroid=centroid,
            )
        )

    return clusters
