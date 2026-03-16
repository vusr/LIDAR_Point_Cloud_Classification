"""
SORT-style multi-object tracker for LiDAR detections.

Each tracked object uses a 2-D Bird's Eye View (BEV) Kalman filter so that
motion estimation is done in the horizontal plane where LiDAR objects move.

State vector:   [x, y, vx, vy]
Measurement:    [x, y]  (centroid of the cluster projected to the XY plane)

Frame-to-frame assignment uses the Hungarian algorithm with Euclidean
centroid distance as the cost.  Separate ClassTracker instances are kept
per semantic class so that a car can never be matched to a pedestrian.

Usage:
    from tracker import MultiClassTracker
    tracker = MultiClassTracker()
    for frame_idx, detections in enumerate(all_detections):
        tracked = tracker.update(detections, frame_idx)
        # tracked: list of TrackedObject
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------

# Maximum Euclidean XY distance (m) allowed for a match
MAX_MATCH_DIST: dict[str, float] = {
    "car":        5.0,
    "pedestrian": 2.0,
    "bicyclist":  3.0,
}

# How many consecutive frames an unmatched track survives before deletion
MAX_AGE: int = 3

# A track must be confirmed by this many hits before it is returned
MIN_HITS: int = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Public representation of a currently active track."""
    track_id:   int
    class_name: str
    centroid:   np.ndarray    # (3,) — x, y, z in world coordinates
    bbox_min:   np.ndarray    # (3,)
    bbox_max:   np.ndarray    # (3,)
    age:        int           # frames since first detection
    hits:       int           # total times matched
    consecutive_misses: int   # frames without a match
    confidence: float         # classifier confidence at latest detection


# ---------------------------------------------------------------------------
# Kalman filter factory
# ---------------------------------------------------------------------------

def _make_bev_kalman(initial_xy: np.ndarray, dt: float = 0.1) -> KalmanFilter:
    """
    Constant-velocity BEV Kalman filter.

    State:       [x, y, vx, vy]  (4-D)
    Measurement: [x, y]          (2-D)

    Args:
        initial_xy: (2,) initial position in metres.
        dt:         Time step between frames (seconds).  The Waymo
                    optional-challenge frames are ~0.1 s apart.
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State-transition matrix (constant velocity)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
    ], dtype=np.float32)

    # Measurement matrix (observe x, y only)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Measurement noise
    kf.R = np.eye(2, dtype=np.float32) * 0.5

    # Process noise (higher on velocity to allow manoeuvres)
    kf.Q = np.diag([0.1, 0.1, 1.0, 1.0]).astype(np.float32)

    # Initial covariance (high uncertainty on velocity)
    kf.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)

    # Initial state
    kf.x = np.array([[initial_xy[0]], [initial_xy[1]], [0.0], [0.0]], dtype=np.float32)

    return kf


# ---------------------------------------------------------------------------
# Single track
# ---------------------------------------------------------------------------

class _KalmanTrack:
    """Internal: one Kalman-filter-backed track."""

    _id_counter: int = 0

    def __init__(
        self,
        class_name: str,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        confidence: float,
    ):
        _KalmanTrack._id_counter += 1
        self.track_id:  int   = _KalmanTrack._id_counter
        self.class_name: str  = class_name
        self.kf:  KalmanFilter = _make_bev_kalman(centroid[:2])
        self.z_pos:   float   = float(centroid[2])   # keep z separately
        self.bbox_min: np.ndarray = bbox_min.copy()
        self.bbox_max: np.ndarray = bbox_max.copy()
        self.confidence: float    = confidence
        self.age: int             = 1
        self.hits: int            = 1
        self.consecutive_misses: int = 0

    @property
    def predicted_xy(self) -> np.ndarray:
        """Current predicted XY position from the Kalman state."""
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0]], dtype=np.float32)

    def predict(self):
        """Advance Kalman prediction by one time step."""
        self.kf.predict()
        self.age += 1
        self.consecutive_misses += 1

    def update(
        self,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        confidence: float,
    ):
        """Incorporate a new matched measurement."""
        z = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kf.update(z)
        self.z_pos      = float(centroid[2])
        self.bbox_min   = bbox_min.copy()
        self.bbox_max   = bbox_max.copy()
        self.confidence = confidence
        self.hits      += 1
        self.consecutive_misses = 0

    def to_tracked_object(self) -> TrackedObject:
        xy = self.predicted_xy
        centroid = np.array([xy[0], xy[1], self.z_pos], dtype=np.float32)
        return TrackedObject(
            track_id=self.track_id,
            class_name=self.class_name,
            centroid=centroid,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            age=self.age,
            hits=self.hits,
            consecutive_misses=self.consecutive_misses,
            confidence=self.confidence,
        )


# ---------------------------------------------------------------------------
# Per-class tracker
# ---------------------------------------------------------------------------

class ClassTracker:
    """
    Manages active tracks for a single semantic class.

    Args:
        class_name:     One of "car", "pedestrian", "bicyclist".
        max_age:        Frames before an unmatched track is dropped.
        min_hits:       Minimum matches before a track is returned.
        max_match_dist: Maximum centroid distance (m) for a valid match.
    """

    def __init__(
        self,
        class_name: str,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        max_match_dist: Optional[float] = None,
    ):
        self.class_name     = class_name
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.max_match_dist = max_match_dist or MAX_MATCH_DIST.get(class_name, 3.0)
        self.tracks: list[_KalmanTrack] = []

    def update(
        self,
        detections: list[dict],
    ) -> list[TrackedObject]:
        """
        Update tracks with new detections for this frame.

        Args:
            detections: List of dicts, each with keys:
                          centroid  (3,)
                          bbox_min  (3,)
                          bbox_max  (3,)
                          confidence float

        Returns:
            List of active TrackedObject instances (confirmed tracks only).
        """
        # 1. Predict all existing tracks
        for t in self.tracks:
            t.predict()

        # 2. Match detections → tracks via Hungarian on XY centroid distance
        matched_track_ids, matched_det_ids = set(), set()

        if self.tracks and detections:
            track_xy = np.stack([t.predicted_xy for t in self.tracks])  # (T, 2)
            det_xy   = np.stack([d["centroid"][:2] for d in detections])  # (D, 2)

            cost = cdist(track_xy, det_xy, metric="euclidean")  # (T, D)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_match_dist:
                    self.tracks[r].update(
                        centroid=detections[c]["centroid"],
                        bbox_min=detections[c]["bbox_min"],
                        bbox_max=detections[c]["bbox_max"],
                        confidence=detections[c]["confidence"],
                    )
                    matched_track_ids.add(r)
                    matched_det_ids.add(c)

        # 3. Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_ids:
                self.tracks.append(
                    _KalmanTrack(
                        class_name=self.class_name,
                        centroid=det["centroid"],
                        bbox_min=det["bbox_min"],
                        bbox_max=det["bbox_max"],
                        confidence=det["confidence"],
                    )
                )

        # 4. Remove stale tracks
        self.tracks = [
            t for t in self.tracks
            if t.consecutive_misses <= self.max_age
        ]

        # 5. Return confirmed tracks
        return [
            t.to_tracked_object()
            for t in self.tracks
            if t.hits >= self.min_hits
        ]


# ---------------------------------------------------------------------------
# Multi-class tracker (public API)
# ---------------------------------------------------------------------------

@dataclass
class FrameDetection:
    """One classified cluster ready for the tracker."""
    class_name: str
    centroid:   np.ndarray    # (3,)
    bbox_min:   np.ndarray    # (3,)
    bbox_max:   np.ndarray    # (3,)
    confidence: float


class MultiClassTracker:
    """
    Wraps one ClassTracker per trackable class.

    Usage:
        tracker = MultiClassTracker()
        for frame_dets in all_frames:
            tracked = tracker.update(frame_dets)
    """

    TRACKABLE_CLASSES: tuple[str, ...] = ("car", "pedestrian", "bicyclist")

    def __init__(self, max_age: int = MAX_AGE, min_hits: int = MIN_HITS):
        self._class_trackers: dict[str, ClassTracker] = {
            cls: ClassTracker(cls, max_age=max_age, min_hits=min_hits)
            for cls in self.TRACKABLE_CLASSES
        }

    @classmethod
    def reset_ids(cls):
        """Reset global track-ID counter (useful between pipeline runs)."""
        _KalmanTrack._id_counter = 0

    def update(self, detections: list[FrameDetection]) -> list[TrackedObject]:
        """
        Process one frame's detections and return all active tracks.

        Args:
            detections: List of FrameDetection for the current frame.

        Returns:
            Flat list of TrackedObject across all classes.
        """
        # Group by class
        by_class: dict[str, list[dict]] = {c: [] for c in self.TRACKABLE_CLASSES}
        for det in detections:
            if det.class_name in by_class:
                by_class[det.class_name].append({
                    "centroid":   det.centroid,
                    "bbox_min":   det.bbox_min,
                    "bbox_max":   det.bbox_max,
                    "confidence": det.confidence,
                })

        all_tracked: list[TrackedObject] = []
        for cls, cls_dets in by_class.items():
            all_tracked.extend(self._class_trackers[cls].update(cls_dets))

        return all_tracked
