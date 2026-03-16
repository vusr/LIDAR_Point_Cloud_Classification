"""
Ground segmentation for full LiDAR scenes using RANSAC plane fitting.

The dominant ground plane is estimated with a pure NumPy RANSAC implementation
(no open3d dependency).  Three random points are sampled per iteration to form a
plane hypothesis; the hypothesis with the most inliers wins.

Usage (standalone):
    from ground_removal import segment_ground
    ground_mask = segment_ground(points_xyz)   # (N,) bool
"""

import numpy as np


def _fit_plane(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """
    Fit a plane through three 3-D points.

    Returns:
        (a, b, c, d) such that  ax + by + cz + d = 0,  normalised so that
        sqrt(a²+b²+c²) = 1.  Returns None if the three points are collinear.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    a, b, c = normal
    d = -np.dot(normal, p1)
    return a, b, c, d


def _ransac_plane(
    pts: np.ndarray,
    num_iterations: int,
    distance_threshold: float,
    rng: np.random.Generator,
) -> tuple:
    """
    RANSAC loop: returns the best (a, b, c, d) plane found.

    Args:
        pts:                (M, 3) candidate seed points.
        num_iterations:     Number of random hypotheses to try.
        distance_threshold: Inlier threshold in metres.
        rng:                NumPy random generator for reproducibility.

    Returns:
        Best (a, b, c, d) plane coefficients (unit normal).
    """
    best_plane = None
    best_inliers = -1
    n = len(pts)

    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        plane = _fit_plane(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if plane is None:
            continue
        a, b, c, d = plane
        dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
        inliers = int((dist < distance_threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = plane

    return best_plane


def segment_ground(
    points_xyz: np.ndarray,
    distance_threshold: float = 0.25,
    num_iterations: int = 200,
    max_ground_z: float = 0.5,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Identify ground points via RANSAC plane fitting (pure NumPy).

    Strategy:
        1. Restrict the RANSAC seed set to low-z points (below `max_ground_z`)
           so that the dominant plane found is the road, not a building wall.
        2. Label all points within `distance_threshold` of that plane as ground.

    Args:
        points_xyz:         (N, 3) float32 — x, y, z coordinates only.
        distance_threshold: Max distance from the fitted plane to be called
                            ground (metres).
        num_iterations:     RANSAC iterations (more → more robust, slower).
        max_ground_z:       Points with z > this value are excluded from the
                            seed cloud so that elevated structures don't confuse
                            RANSAC.
        random_seed:        Seed for reproducible sampling.

    Returns:
        ground_mask: (N,) bool — True where a point is classified as ground.
    """
    rng = np.random.default_rng(random_seed)

    # --- Seed cloud: only low-z points to bias the plane towards the road ---
    seed_mask = points_xyz[:, 2] < max_ground_z
    seed_xyz  = points_xyz[seed_mask]

    if len(seed_xyz) < 3:
        seed_xyz = points_xyz

    plane = _ransac_plane(
        seed_xyz.astype(np.float64),
        num_iterations=num_iterations,
        distance_threshold=distance_threshold,
        rng=rng,
    )

    if plane is None:
        return np.zeros(len(points_xyz), dtype=bool)

    a, b, c, d = plane

    # Signed distance of every point to the estimated plane (normal is unit)
    signed_dist = (
        a * points_xyz[:, 0]
        + b * points_xyz[:, 1]
        + c * points_xyz[:, 2]
        + d
    )

    ground_mask = np.abs(signed_dist) < distance_threshold

    # Extra guard: if the plane normal points mostly upward (|c| large), points
    # clearly above the plane cannot be ground.
    if abs(c) > 0.5:
        ground_mask[signed_dist > distance_threshold] = False

    return ground_mask
