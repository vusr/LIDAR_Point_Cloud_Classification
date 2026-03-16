"""
LiDAR point cloud ↔ range image conversion for SalsaNext inference.

The optional-challenge data has 5 channels per point:
    col 0: x          (metres)
    col 1: y          (metres)
    col 2: z          (metres)
    col 3: intensity  (0–1 float)
    col 4: ring       (0–63 integer scan-line index)

SalsaNext expects a (B, 5, H, W) range image with channels:
    0: range      (Euclidean distance from sensor)
    1: x
    2: y
    3: z
    4: remission  (intensity)

normalised with the SemanticKITTI sensor statistics so that the pretrained
weights interpret the input correctly.

Key design decision: the ring channel is used directly as the row index (H axis)
which avoids computing elevation angles and gives a perfect 1-to-1 mapping
between scan lines and image rows.

Public API:
    project_to_range_image(points, H, W)  →  (range_image, index_map)
    unproject_labels(label_image, index_map, N)  →  per_point_labels
"""

import numpy as np

# ---------------------------------------------------------------------------
# SemanticKITTI normalisation statistics
# (computed over the full training split; used to match the pretrained weights)
# ---------------------------------------------------------------------------
_MEANS = np.array([11.71279, -0.1023,  0.4916,  -1.0795, 0.2702], dtype=np.float32)
_STDS  = np.array([10.24,    12.295,   9.4,      0.9,    0.1355], dtype=np.float32)

# Default range image dimensions matching a 64-beam, 2048-col LiDAR
DEFAULT_H: int = 64
DEFAULT_W: int = 2048


def project_to_range_image(
    points: np.ndarray,
    H: int = DEFAULT_H,
    W: int = DEFAULT_W,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a LiDAR point cloud into a (H, W) range image.

    For each point:
      - Row  = ring index (points[:, 4])
      - Col  = azimuth angle mapped to [0, W)

    When multiple points map to the same pixel the closest one (smallest range)
    is kept so that foreground objects are not occluded by distant background.

    Args:
        points: (N, 5) float32 array — x, y, z, intensity, ring.
        H:      Number of rows (= number of LiDAR beams / rings).
        W:      Number of columns (horizontal resolution).

    Returns:
        range_image: (5, H, W) float32 normalised range image ready for
                     SalsaNext inference.  Zero-filled for empty pixels.
        index_map:   (H, W) int32 array — original point index at each pixel,
                     or -1 for empty pixels.
    """
    x         = points[:, 0]
    y         = points[:, 1]
    z         = points[:, 2]
    intensity = points[:, 3]
    rings     = points[:, 4].astype(np.int32)

    # Clip rings to valid range to guard against sensor artefacts
    rings = np.clip(rings, 0, H - 1)

    # Range (Euclidean distance)
    range_vals = np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(np.float32)

    # Azimuth → column index
    yaw = np.arctan2(y, x)                          # (-π, π)
    col = ((yaw + np.pi) / (2.0 * np.pi) * W)       # (0, W)
    col = np.clip(col.astype(np.int32), 0, W - 1)

    # Build range image (5 channels) and index map
    range_image = np.zeros((5, H, W), dtype=np.float32)
    index_map   = np.full((H, W), -1, dtype=np.int32)

    # Fill in order of decreasing range so that the closest point wins
    # (overwrite only when the new point is closer)
    order = np.argsort(-range_vals)   # farthest first → closest overwrites

    for idx in order:
        r = rings[idx]
        c = col[idx]
        range_image[0, r, c] = range_vals[idx]
        range_image[1, r, c] = x[idx]
        range_image[2, r, c] = y[idx]
        range_image[3, r, c] = z[idx]
        range_image[4, r, c] = intensity[idx]
        index_map[r, c]      = idx

    # Normalise with SemanticKITTI statistics
    # range_image shape: (5, H, W); means/stds shape: (5,) → reshape to (5, 1, 1)
    means = _MEANS[:, None, None]
    stds  = _STDS[:, None, None]

    # Only normalise occupied pixels (leave empty pixels at 0)
    occupied = index_map >= 0   # (H, W) bool
    for ch in range(5):
        ch_slice = range_image[ch]
        ch_slice[occupied] = (ch_slice[occupied] - means[ch, 0, 0]) / (stds[ch, 0, 0] + 1e-8)

    return range_image, index_map


def unproject_labels(
    label_image: np.ndarray,
    index_map:   np.ndarray,
    N:           int,
) -> np.ndarray:
    """
    Map a (H, W) per-pixel label image back to a flat per-point label array.

    Points that landed on an empty pixel or were overwritten by a closer point
    receive label 0 (background / unlabeled).

    Args:
        label_image: (H, W) int32 — argmax output from SalsaNext.
        index_map:   (H, W) int32 — from project_to_range_image; -1 = empty.
        N:           Total number of original points.

    Returns:
        per_point_labels: (N,) int32 — SalsaNext class index per point.
    """
    per_point_labels = np.zeros(N, dtype=np.int32)

    valid = index_map >= 0          # (H, W) bool
    pts   = index_map[valid]        # point indices that own a pixel
    lbls  = label_image[valid]      # their labels

    per_point_labels[pts] = lbls
    return per_point_labels
