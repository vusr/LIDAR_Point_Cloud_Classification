"""
PointNet++ (Single-Scale Grouping) for point cloud classification.

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space", NeurIPS 2017.

Architecture overview:
    SA1 (local)  →  SA2 (local)  →  SA3 (global)  →  FC classifier

Each Set Abstraction (SA) layer:
    1. Farthest Point Sampling  → select npoint centroids
    2. Ball Query               → group neighbours within radius
    3. PointNet (shared MLP)    → per-group feature extraction
    4. Max pooling              → aggregate group features

The final SA layer uses all remaining points as a single group
(i.e., global pooling) to produce a fixed-length feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Iterative farthest point sampling (FPS).

    Args:
        xyz:    (B, N, 3) point coordinates
        npoint: number of centroids to sample

    Returns:
        idx: (B, npoint) indices of sampled points in [0, N)
    """
    B, N, _ = xyz.shape
    device   = xyz.device

    centroids  = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances  = torch.full((B, N), float("inf"), device=device)
    # Start from a random point per batch element
    farthest   = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        # (B, 3) — current centroid coordinates
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        # Squared distances from all points to the current centroid
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        # Update minimum distances
        distances = torch.min(distances, dist)
        # Pick the farthest point
        farthest = distances.max(dim=1)[1]

    return centroids


def index_points(pts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.

    Args:
        pts: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        gathered: same shape as idx but with C-dim appended
    """
    B = pts.shape[0]
    view_shape  = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (
        torch.arange(B, dtype=torch.long, device=pts.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return pts[batch_idx, idx, :]


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    For each centroid in new_xyz, find up to nsample neighbours within `radius`
    from xyz.  Missing neighbours are filled with the centroid's own index.

    Args:
        radius:  neighbourhood radius
        nsample: maximum number of neighbours
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) centroids

    Returns:
        group_idx: (B, S, nsample) indices into xyz
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device   = xyz.device

    # Pairwise squared distances: (B, S, N)
    sq_dists = (
        torch.sum(new_xyz ** 2, dim=-1, keepdim=True)           # (B, S, 1)
        + torch.sum(xyz ** 2, dim=-1).unsqueeze(1)               # (B, 1, N)
        - 2 * torch.bmm(new_xyz, xyz.transpose(1, 2))           # (B, S, N)
    )

    # Replace out-of-radius indices with N (sentinel)
    group_idx = sq_dists.argsort(dim=-1)[..., :nsample]          # (B, S, nsample)
    # Mask points that fall outside the radius
    mask = sq_dists.gather(dim=-1, index=group_idx) > radius ** 2
    # Fill masked positions with the centroid's own index (position 0 = nearest)
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    group_idx[mask] = group_first[mask]

    return group_idx


# ---------------------------------------------------------------------------
# Shared MLP building block
# ---------------------------------------------------------------------------

def build_mlp(channels: list[int], bn: bool = True) -> nn.Sequential:
    """1-D convolution MLP (operates on last dimension via Conv1d/Conv2d)."""
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Set Abstraction Layer
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    """
    Single-Scale Grouping Set Abstraction layer.

    Args:
        npoint:   number of centroids to sample via FPS
        radius:   ball query radius
        nsample:  max neighbours per centroid
        in_ch:    input channel count (NOT counting xyz, which is concatenated)
        mlp_ch:   list of output channels for the per-group MLP
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_ch: int,
        mlp_ch: list[int],
    ):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample
        # Input to MLP: 3 (relative xyz) + in_ch features
        self.mlp = build_mlp([3 + in_ch] + mlp_ch)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, N, C) or None for the first SA layer

        Returns:
            new_xyz:      (B, npoint, 3)
            new_features: (B, npoint, mlp_ch[-1])
        """
        B, N, _ = xyz.shape

        # 1. FPS → centroids
        fps_idx  = farthest_point_sample(xyz, self.npoint)  # (B, S)
        new_xyz  = index_points(xyz, fps_idx)                # (B, S, 3)

        # 2. Ball query → neighbour indices
        group_idx   = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, S, K)
        # Out-of-place subtraction avoids in-place op on non-contiguous tensor
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)  # (B, S, K, 3)

        if features is not None:
            grouped_feat = index_points(features, group_idx)                 # (B, S, K, C)
            grouped_pts  = torch.cat([grouped_xyz, grouped_feat], dim=-1)    # (B, S, K, 3+C)
        else:
            grouped_pts = grouped_xyz                                         # (B, S, K, 3)

        # 3. MLP — Conv2d expects (B, C, S, K); .contiguous() required after permute
        grouped_pts  = grouped_pts.permute(0, 3, 1, 2).contiguous()          # (B, 3+C, S, K)
        grouped_pts  = self.mlp(grouped_pts)                                  # (B, D, S, K)

        # 4. Max pool over neighbours; .contiguous() required after permute
        new_features = grouped_pts.max(dim=-1)[0].permute(0, 2, 1).contiguous()  # (B, S, D)

        return new_xyz, new_features


class GlobalSetAbstraction(nn.Module):
    """
    Global Set Abstraction — uses all remaining points as a single group,
    producing a single fixed-length feature vector per batch element.

    Args:
        in_ch:   number of input feature channels (NOT counting xyz)
        mlp_ch:  list of output channels for the MLP
    """

    def __init__(self, in_ch: int, mlp_ch: list[int]):
        super().__init__()
        # 1-D MLP (no grouping, so Conv1d is sufficient)
        layers = []
        prev = 3 + in_ch
        for ch in mlp_ch:
            layers += [
                nn.Conv1d(prev, ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, N, C)

        Returns:
            global_feat: (B, mlp_ch[-1])
        """
        # Concatenate xyz and features, transpose to (B, 3+C, N); .contiguous() required after permute
        combined = torch.cat([xyz, features], dim=-1).permute(0, 2, 1).contiguous()
        out = self.mlp(combined)               # (B, D, N)
        global_feat = out.max(dim=-1)[0]       # (B, D)
        return global_feat


# ---------------------------------------------------------------------------
# PointNet++ Classifier
# ---------------------------------------------------------------------------

class PointNet2Classifier(nn.Module):
    """
    PointNet++ (SSG) for point cloud object classification.

    Default architecture is tuned for small outdoor LiDAR clusters
    (cars, pedestrians, bicyclists) that span roughly 0.5 – 8 m and
    contain a variable number of points (typically 50 – 1000).

    Args:
        num_classes:     number of output classes
        sa1_npoint:      FPS centroids for SA1
        sa1_radius:      ball query radius for SA1
        sa1_nsample:     max neighbours for SA1
        sa2_npoint:      FPS centroids for SA2
        sa2_radius:      ball query radius for SA2
        sa2_nsample:     max neighbours for SA2
        dropout_rate:    dropout probability before final FC
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        sa1_npoint:   int   = 128,
        sa1_radius:   float = 0.3,
        sa1_nsample:  int   = 32,
        sa2_npoint:   int   = 32,
        sa2_radius:   float = 0.6,
        sa2_nsample:  int   = 64,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # SA1: 3 → 128-dim features
        self.sa1 = SetAbstraction(
            npoint=sa1_npoint,
            radius=sa1_radius,
            nsample=sa1_nsample,
            in_ch=0,
            mlp_ch=[64, 64, 128],
        )

        # SA2: 128 → 256-dim features
        self.sa2 = SetAbstraction(
            npoint=sa2_npoint,
            radius=sa2_radius,
            nsample=sa2_nsample,
            in_ch=128,
            mlp_ch=[128, 128, 256],
        )

        # SA3: global → 1024-dim feature vector
        self.sa3 = GlobalSetAbstraction(
            in_ch=256,
            mlp_ch=[256, 512, 1024],
        )

        # FC head
        self.fc1      = nn.Linear(1024, 512)
        self.bn1      = nn.BatchNorm1d(512)
        self.drop1    = nn.Dropout(dropout_rate)

        self.fc2      = nn.Linear(512, 256)
        self.bn2      = nn.BatchNorm1d(256)
        self.drop2    = nn.Dropout(dropout_rate)

        self.fc3      = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)  raw point cloud (already normalised)

        Returns:
            logits: (B, num_classes)
        """
        # xyz passed as (B, N, 3); no initial per-point features
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        global_feat = self.sa3(xyz2, feat2)

        x = self.drop1(F.relu(self.bn1(self.fc1(global_feat)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        logits = self.fc3(x)
        return logits
