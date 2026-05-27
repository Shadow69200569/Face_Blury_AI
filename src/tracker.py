"""
Face Tracking Module — Simplified ByteTrack Implementation
===========================================================

Maintains consistent face identities across video frames using the
ByteTrack algorithm (Zhang et al., ECCV 2022). This is critical for:
    1. Smooth, flicker-free blur — same face keeps same blur region
    2. Frame-skip optimisation — tracker predicts positions on skipped frames
    3. Temporal smoothing — EMA across frames requires stable IDs

Architecture:
    KalmanBoxTracker — Per-target Kalman filter for motion prediction
    Track            — Lifecycle management (tentative → confirmed → lost)
    ByteTracker      — Two-stage association using IoU + Hungarian algorithm
    TemporalSmoother — EMA-based bounding box stabilisation

Engineering Decisions:
    - filterpy is preferred for Kalman filtering but optional; a lightweight
      EMA fallback is implemented so the tracker works zero-dependency.
    - scipy.optimize.linear_sum_assignment is used for Hungarian matching;
      if unavailable, a greedy matcher provides a reasonable approximation.
    - The two-stage association (ByteTrack's core idea) recovers partially
      visible / low-confidence faces that single-stage trackers would miss.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import FaceBlurConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FaceDetection import — must match the canonical definition in detector.py.
# Since detector.py might not exist yet at import time, we define a local
# copy that is structurally identical.  The video_pipeline module will treat
# them interchangeably because they share the same field names.
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """Single face detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence in [0, 1].
        landmarks: Optional 5×2 array of facial landmarks
                   (left-eye, right-eye, nose, left-mouth, right-mouth).
        track_id: Assigned tracker identity (-1 = untracked).
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    track_id: int = -1


# ---------------------------------------------------------------------------
# Kalman Filter — motion model for a single tracked bounding box
# ---------------------------------------------------------------------------

# Try importing filterpy; fall back to a minimal EMA tracker if unavailable.
_HAS_FILTERPY = False
try:
    from filterpy.kalman import KalmanFilter as _FilterpyKF
    _HAS_FILTERPY = True
    logger.debug("filterpy available — using full Kalman filter for tracking")
except ImportError:
    logger.info(
        "filterpy not installed — falling back to EMA-based motion model. "
        "Install filterpy for better tracking: pip install filterpy"
    )


def _bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Convert (x1, y1, x2, y2) bbox to Kalman measurement [cx, cy, area, ratio].

    Using (center_x, center_y, area, aspect_ratio) as the measurement space
    decouples position from scale, which makes the constant-velocity model
    work better for faces that grow/shrink as a person approaches/recedes.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    area = w * h
    ratio = w / max(h, 1e-6)
    return np.array([cx, cy, area, ratio], dtype=np.float64)


def _z_to_bbox(z: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert Kalman state [cx, cy, area, ratio, ...] back to (x1, y1, x2, y2).

    Clamps area to positive to avoid sqrt-of-negative after noisy predictions.
    """
    cx, cy = z[0], z[1]
    area = max(z[2], 1.0)
    ratio = max(z[3], 0.01)  # prevent degenerate aspect ratio
    w = np.sqrt(area * ratio)
    h = area / max(w, 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


class KalmanBoxTracker:
    """Per-target Kalman filter wrapping a 7-D constant-velocity model.

    State vector: [cx, cy, area, ratio, v_cx, v_cy, v_area]
    Measurement:  [cx, cy, area, ratio]

    If filterpy is unavailable, transparently falls back to a simple
    exponential moving average (EMA) tracker that has no motion model
    but is sufficient for moderate frame rates.

    Args:
        bbox: Initial bounding box (x1, y1, x2, y2).
        ema_alpha: Smoothing factor for the EMA fallback (0 = no update, 1 = no smoothing).
    """

    _count: int = 0  # class-level ID counter

    def __init__(self, bbox: Tuple[int, int, int, int], ema_alpha: float = 0.7) -> None:
        self._use_kalman = _HAS_FILTERPY
        self._ema_alpha = ema_alpha

        if self._use_kalman:
            self._init_kalman(bbox)
        else:
            # EMA fallback — just store the measurement vector
            self._state = _bbox_to_z(bbox).astype(np.float64)

    # ---- filterpy Kalman initialisation ------------------------------------

    def _init_kalman(self, bbox: Tuple[int, int, int, int]) -> None:
        """Set up a 7-D constant-velocity Kalman filter.

        The matrices follow the standard ByteTrack / SORT parameterisation.
        Measurement noise is tuned for face-sized objects at typical
        webcam / surveillance resolutions (480p–1080p).
        """
        self._kf = _FilterpyKF(dim_x=7, dim_z=4)

        # State transition: constant velocity
        # x_{k+1} = F @ x_k
        self._kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix: we observe [cx, cy, area, ratio]
        self._kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Measurement noise — relatively low because face detectors are
        # fairly precise on position; area noise is higher because
        # bounding box size fluctuates more.
        self._kf.R[2:, 2:] *= 10.0

        # Initial covariance — high uncertainty on velocities
        self._kf.P[4:, 4:] *= 1000.0
        self._kf.P *= 10.0

        # Process noise — allow moderate acceleration
        self._kf.Q[-1, -1] *= 0.01
        self._kf.Q[4:, 4:] *= 0.01

        # Initialise state from first measurement
        z = _bbox_to_z(bbox)
        self._kf.x[:4] = z.reshape((4, 1))

    # ---- Public API --------------------------------------------------------

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance the state by one time-step and return predicted bbox.

        For the EMA fallback, prediction is a no-op (last state persists).
        """
        if self._use_kalman:
            self._kf.predict()
            return _z_to_bbox(self._kf.x.flatten())
        else:
            # EMA has no motion model — just return current state
            return _z_to_bbox(self._state)

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Incorporate a new detection measurement.

        Args:
            bbox: Observed bounding box (x1, y1, x2, y2).
        """
        z = _bbox_to_z(bbox)
        if self._use_kalman:
            self._kf.update(z.reshape((4, 1)))
        else:
            # EMA update: state = alpha * measurement + (1 - alpha) * state
            self._state = self._ema_alpha * z + (1.0 - self._ema_alpha) * self._state

    def get_state(self) -> Tuple[int, int, int, int]:
        """Return the current bbox estimate without advancing time."""
        if self._use_kalman:
            return _z_to_bbox(self._kf.x.flatten())
        else:
            return _z_to_bbox(self._state)


# ---------------------------------------------------------------------------
# Track — lifecycle wrapper around KalmanBoxTracker
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Lifecycle-managed tracked face.

    State machine:
        tentative  → confirmed  (after `_confirm_hits` consecutive matches)
        confirmed  → lost       (after `time_since_update` > track_buffer)
        lost       → deleted    (by ByteTracker pruning)

    Attributes:
        track_id: Unique identity for this face across frames.
        kalman: Underlying motion model.
        hits: Consecutive successful detection matches.
        age: Total number of frames this track has existed.
        time_since_update: Frames elapsed since last matched detection.
        state: Current lifecycle state.
        confidence: Confidence of the most recent matched detection.
    """
    track_id: int
    kalman: KalmanBoxTracker
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "tentative"
    confidence: float = 0.0

    # Internal: number of consecutive hits required for confirmation
    _confirm_hits: int = field(default=3, repr=False)

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance one frame and return predicted bbox."""
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()

    def update(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Update track with a matched detection."""
        self.kalman.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence

        # Promote tentative → confirmed after enough consecutive hits
        if self.state == "tentative" and self.hits >= self._confirm_hits:
            self.state = "confirmed"
            logger.debug("Track %d confirmed after %d hits", self.track_id, self.hits)

    def mark_lost(self) -> None:
        """Transition confirmed → lost."""
        if self.state == "confirmed":
            self.state = "lost"
            logger.debug("Track %d marked lost", self.track_id)

    def is_expired(self, max_age: int) -> bool:
        """Check whether this track should be deleted."""
        return self.time_since_update > max_age

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Current best-estimate bounding box."""
        return self.kalman.get_state()


# ---------------------------------------------------------------------------
# IoU computation — vectorised for speed
# ---------------------------------------------------------------------------

def _compute_iou_matrix(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array of boxes in (x1, y1, x2, y2) format.
        bboxes_b: (M, 4) array of boxes in (x1, y1, x2, y2) format.

    Returns:
        (N, M) IoU matrix.
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.empty((len(bboxes_a), len(bboxes_b)), dtype=np.float64)

    # Broadcast intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    # Avoid division by zero for degenerate boxes
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Hungarian / Linear Assignment wrapper
# ---------------------------------------------------------------------------

# Try scipy first; fall back to a greedy matcher.
_HAS_SCIPY = False
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except ImportError:
    logger.info(
        "scipy not installed — using greedy assignment fallback. "
        "Install scipy for optimal track matching: pip install scipy"
    )


def _linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve the linear assignment problem on a cost matrix.

    Uses scipy's Hungarian algorithm when available, otherwise a greedy
    approach that picks the best remaining match iteratively.

    Args:
        cost_matrix: (N, M) cost matrix where LOWER is better.
                     Here we use (1 - IoU) so 0 = perfect overlap.
        threshold: Maximum cost to accept a match.

    Returns:
        matched: List of (row, col) matched index pairs.
        unmatched_rows: Row indices with no valid match.
        unmatched_cols: Column indices with no valid match.
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    if _HAS_SCIPY:
        row_idx, col_idx = _scipy_lsa(cost_matrix)
    else:
        # Greedy fallback — O(N*M) per match, acceptable for small N, M
        row_idx, col_idx = [], []
        cost_copy = cost_matrix.copy()
        used_rows: set = set()
        used_cols: set = set()
        for _ in range(min(cost_matrix.shape)):
            # Mask already-used entries
            cost_copy[list(used_rows), :] = np.inf
            cost_copy[:, list(used_cols)] = np.inf
            idx = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
            if cost_copy[idx] == np.inf:
                break
            row_idx.append(idx[0])
            col_idx.append(idx[1])
            used_rows.add(idx[0])
            used_cols.add(idx[1])
        row_idx = np.array(row_idx, dtype=int)
        col_idx = np.array(col_idx, dtype=int)

    # Filter matches by threshold
    matched: List[Tuple[int, int]] = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= threshold:
            matched.append((int(r), int(c)))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matched, sorted(unmatched_rows), sorted(unmatched_cols)


# ---------------------------------------------------------------------------
# ByteTracker — main tracking engine
# ---------------------------------------------------------------------------

class ByteTracker:
    """Simplified ByteTrack algorithm for face tracking.

    Two-stage association is the key idea:
        Stage 1: Match high-confidence detections to tracks using IoU.
        Stage 2: Match remaining LOW-confidence detections to unmatched
                 tracks. This recovers partially occluded or motion-blurred
                 faces that would be lost by traditional trackers.

    Args:
        config: Pipeline configuration with tracking parameters.
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        self._config = config
        self._tracks: List[Track] = []
        self._next_id: int = 1  # start IDs from 1 (0 is sometimes sentinel)
        self._frame_count: int = 0

        # Cache config values for hot-path access
        self._track_buffer = config.track_buffer
        self._match_threshold = config.match_threshold
        self._high_thresh = config.track_high_thresh
        self._low_thresh = config.track_low_thresh

        logger.info(
            "ByteTracker initialised — buffer=%d, match_thresh=%.2f, "
            "high_thresh=%.2f, low_thresh=%.2f",
            self._track_buffer,
            self._match_threshold,
            self._high_thresh,
            self._low_thresh,
        )

    # ---- Public API --------------------------------------------------------

    def update(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Run one frame of the ByteTrack algorithm.

        Assigns stable track_id values to each detection, creating,
        updating, or removing tracks as necessary.

        Args:
            detections: Raw face detections for the current frame.

        Returns:
            Detections with track_id populated (only confirmed tracks).
        """
        self._frame_count += 1

        # ── Step 0: predict new positions for all existing tracks ──
        for track in self._tracks:
            track.predict()

        # ── Step 1: split detections by confidence ──
        det_high: List[int] = []
        det_low: List[int] = []
        for i, det in enumerate(detections):
            if det.confidence >= self._high_thresh:
                det_high.append(i)
            elif det.confidence >= self._low_thresh:
                det_low.append(i)
            # Detections below low_thresh are discarded entirely

        # ── Step 2: first association — high-confidence vs ALL tracks ──
        confirmed_tracks = [
            i for i, t in enumerate(self._tracks)
            if t.state in ("confirmed", "tentative")
        ]

        matched_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            track_indices=confirmed_tracks,
            det_indices=det_high,
            detections=detections,
        )

        # Apply first-round matches
        for t_idx, d_idx in matched_1:
            det = detections[d_idx]
            self._tracks[t_idx].update(det.bbox, det.confidence)

        # ── Step 3: second association — low-confidence vs remaining tracks ──
        remaining_tracks = unmatched_tracks_1
        matched_2, unmatched_tracks_2, unmatched_dets_2 = self._associate(
            track_indices=remaining_tracks,
            det_indices=det_low,
            detections=detections,
        )

        # Apply second-round matches
        for t_idx, d_idx in matched_2:
            det = detections[d_idx]
            self._tracks[t_idx].update(det.bbox, det.confidence)

        # ── Step 4: handle unmatched tracks — mark lost or delete ──
        for t_idx in unmatched_tracks_2:
            track = self._tracks[t_idx]
            track.mark_lost()

        # ── Step 5: create new tracks for unmatched HIGH-confidence detections ──
        for d_idx in unmatched_dets_1:
            det = detections[d_idx]
            new_track = Track(
                track_id=self._next_id,
                kalman=KalmanBoxTracker(det.bbox),
                confidence=det.confidence,
            )
            self._next_id += 1
            self._tracks.append(new_track)
            logger.debug(
                "New track %d created at bbox=%s conf=%.2f",
                new_track.track_id, det.bbox, det.confidence,
            )

        # ── Step 6: prune expired tracks ──
        active_tracks: List[Track] = []
        for track in self._tracks:
            if not track.is_expired(self._track_buffer):
                active_tracks.append(track)
            else:
                logger.debug("Track %d deleted (expired)", track.track_id)
        self._tracks = active_tracks

        # ── Step 7: build output — assign track_ids to detections ──
        output: List[FaceDetection] = []
        for track in self._tracks:
            # Only emit confirmed or recently-active tracks
            if track.state in ("confirmed", "tentative") and track.time_since_update == 0:
                bbox = track.get_bbox()
                # Find the detection that was matched to this track
                det = self._find_matched_detection(track, detections)
                output.append(FaceDetection(
                    bbox=bbox,
                    confidence=track.confidence,
                    landmarks=det.landmarks if det else None,
                    track_id=track.track_id,
                ))

        logger.debug(
            "Frame %d — %d detections in, %d tracked out, %d active tracks",
            self._frame_count, len(detections), len(output), len(self._tracks),
        )
        return output

    def reset(self) -> None:
        """Clear all tracks and reset state. Use between video segments."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("ByteTracker reset — all tracks cleared")

    @property
    def active_track_count(self) -> int:
        """Number of currently active (non-expired) tracks."""
        return len(self._tracks)

    # ---- Internal methods --------------------------------------------------

    def _associate(
        self,
        track_indices: List[int],
        det_indices: List[int],
        detections: List[FaceDetection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match tracks to detections via IoU + Hungarian assignment.

        Args:
            track_indices: Indices into self._tracks to consider.
            det_indices: Indices into `detections` to consider.
            detections: Full detection list (indexed by det_indices).

        Returns:
            matched: (track_index, det_index) pairs.
            unmatched_tracks: Track indices with no match.
            unmatched_dets: Detection indices with no match.
        """
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)

        # Build bbox arrays for IoU computation
        track_bboxes = np.array(
            [self._tracks[i].get_bbox() for i in track_indices],
            dtype=np.float64,
        )
        det_bboxes = np.array(
            [detections[i].bbox for i in det_indices],
            dtype=np.float64,
        )

        iou_matrix = _compute_iou_matrix(track_bboxes, det_bboxes)

        # Cost = 1 - IoU  (lower = better match)
        cost_matrix = 1.0 - iou_matrix

        # Threshold: matches with cost > (1 - match_threshold) are rejected
        cost_thresh = 1.0 - self._match_threshold

        matched_local, unmatched_t_local, unmatched_d_local = _linear_assignment(
            cost_matrix, cost_thresh
        )

        # Map local indices back to global indices
        matched_global = [
            (track_indices[t], det_indices[d]) for t, d in matched_local
        ]
        unmatched_tracks_global = [track_indices[i] for i in unmatched_t_local]
        unmatched_dets_global = [det_indices[i] for i in unmatched_d_local]

        return matched_global, unmatched_tracks_global, unmatched_dets_global

    def _iou_matrix(
        self,
        tracks: List[Track],
        detections: List[FaceDetection],
    ) -> np.ndarray:
        """Compute IoU cost matrix between tracks and detections.

        Convenience method exposing the IoU computation for external use
        (e.g. debug overlays, unit tests).

        Args:
            tracks: List of Track objects.
            detections: List of FaceDetection objects.

        Returns:
            (len(tracks), len(detections)) IoU matrix.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float64)

        track_bboxes = np.array(
            [t.get_bbox() for t in tracks], dtype=np.float64
        )
        det_bboxes = np.array(
            [d.bbox for d in detections], dtype=np.float64
        )
        return _compute_iou_matrix(track_bboxes, det_bboxes)

    @staticmethod
    def _find_matched_detection(
        track: Track,
        detections: List[FaceDetection],
    ) -> Optional[FaceDetection]:
        """Find the detection closest to a track's current state.

        Used to carry over landmarks from the raw detection to the
        tracked output. A simple nearest-centre heuristic is sufficient
        because this is only called for already-matched pairs.
        """
        if not detections:
            return None

        t_bbox = track.get_bbox()
        t_cx = (t_bbox[0] + t_bbox[2]) / 2.0
        t_cy = (t_bbox[1] + t_bbox[3]) / 2.0

        best_det = None
        best_dist = float("inf")

        for det in detections:
            d_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            d_cy = (det.bbox[1] + det.bbox[3]) / 2.0
            dist = (t_cx - d_cx) ** 2 + (t_cy - d_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_det = det

        return best_det


# ---------------------------------------------------------------------------
# TemporalSmoother — EMA-based bounding box stabilisation
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """Smooths bounding box positions across frames using exponential moving average.

    Without smoothing, even small frame-to-frame jitter in detection boxes
    causes the blur region to "vibrate", which looks unprofessional.
    EMA smoothing adds a one-frame lag but dramatically reduces jitter.

    Args:
        stale_timeout: Remove history entries for tracks not seen in this
                       many calls to `smooth()`. Prevents unbounded memory
                       growth from long-gone track IDs.
    """

    def __init__(self, stale_timeout: int = 60) -> None:
        # track_id → (smoothed_bbox_array, last_update_frame)
        self._history: Dict[int, Tuple[np.ndarray, int]] = {}
        self._stale_timeout = stale_timeout
        self._call_count: int = 0

    def smooth(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        alpha: float = 0.7,
    ) -> Tuple[int, int, int, int]:
        """Apply EMA smoothing to a tracked bounding box.

        smoothed = alpha * current + (1 - alpha) * previous

        Args:
            track_id: Stable track identity.
            bbox: Raw bounding box (x1, y1, x2, y2).
            alpha: Smoothing factor. Higher = more responsive but jittery.
                   Lower = smoother but laggier. 0.7 is a good default.

        Returns:
            Smoothed bounding box (x1, y1, x2, y2).
        """
        self._call_count += 1
        current = np.array(bbox, dtype=np.float64)

        if track_id in self._history:
            prev, _ = self._history[track_id]
            smoothed = alpha * current + (1.0 - alpha) * prev
        else:
            # First observation — no history to blend with
            smoothed = current

        self._history[track_id] = (smoothed, self._call_count)

        # Periodically prune stale entries to prevent memory leak
        if self._call_count % 100 == 0:
            self._prune_stale()

        return (
            int(round(smoothed[0])),
            int(round(smoothed[1])),
            int(round(smoothed[2])),
            int(round(smoothed[3])),
        )

    def _prune_stale(self) -> None:
        """Remove track histories not updated recently."""
        cutoff = self._call_count - self._stale_timeout
        stale_ids = [
            tid for tid, (_, last_update) in self._history.items()
            if last_update < cutoff
        ]
        for tid in stale_ids:
            del self._history[tid]
        if stale_ids:
            logger.debug(
                "TemporalSmoother pruned %d stale entries", len(stale_ids)
            )

    def reset(self) -> None:
        """Clear all smoothing history."""
        self._history.clear()
        self._call_count = 0
