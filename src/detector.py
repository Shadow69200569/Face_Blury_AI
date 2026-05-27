"""
Face Detection Module for Face Blur Pipeline
==============================================

Core module providing pluggable face detection backends with a unified interface.
All detectors produce standardized ``FaceDetection`` results that downstream
modules (tracker, blur, visualization) consume without knowing which backend
was used.

Architecture:
    FaceDetector (ABC)
    ├── SCRFDDetector       — InsightFace SCRFD (primary, best accuracy)
    ├── RetinaFaceDetector  — InsightFace RetinaFace (alternative)
    ├── YuNetDetector       — OpenCV built-in (zero extra dependencies)
    ├── MediaPipeDetector   — Google BlazeFace (ultra-fast CPU)
    └── EnsembleDetector    — Multi-detector fusion with cross-detector NMS

Engineering Decisions:
    - ABC forces every backend to expose the same `.detect()` signature.
    - Post-processing (NMS, padding, size filter) lives in the base class so
      every backend automatically gets consistent quality-of-life behaviour.
    - ``create_detector`` factory implements a *fallback chain*: if the user
      requests SCRFD but InsightFace is not installed, we fall back to YuNet,
      which ships with OpenCV and requires zero extra packages.
    - IoU-based NMS is implemented from scratch (no torch dependency) to keep
      the core pipeline torch-free.  ONNX Runtime is the only heavy runtime.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import FaceBlurConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """Standardized face detection result.

    Every detection backend maps its native output into this dataclass so that
    the rest of the pipeline (tracker, blur engine, visualization) can consume
    detections uniformly.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in *absolute* pixel coordinates.
              Top-left corner is (x1, y1), bottom-right is (x2, y2).
        confidence: Detection confidence in [0.0, 1.0].
        landmarks: Optional 5×2 array of facial landmark coordinates.
                   Standard order: left-eye, right-eye, nose, left-mouth,
                   right-mouth (following InsightFace convention).
        track_id: Assigned by the tracking stage.  -1 means the detection
                  has not yet been associated with a track.
    """

    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float                  # 0.0 – 1.0
    landmarks: Optional[np.ndarray] = None  # shape (5, 2) or None
    track_id: int = -1                 # -1 = untracked

    # ----- Convenience properties -----------------------------------------

    @property
    def width(self) -> int:
        """Bounding-box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Bounding-box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Bounding-box area in pixels²."""
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """Center (cx, cy) of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def __repr__(self) -> str:
        return (
            f"FaceDetection(bbox={self.bbox}, conf={self.confidence:.3f}, "
            f"track_id={self.track_id})"
        )


# ---------------------------------------------------------------------------
# Geometry / NMS Helpers
# ---------------------------------------------------------------------------

def compute_iou(box_a: Tuple[int, int, int, int],
                box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two axis-aligned boxes.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU value in [0.0, 1.0].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def apply_nms(detections: List[FaceDetection],
              iou_threshold: float = 0.4) -> List[FaceDetection]:
    """Greedy non-maximum suppression on a list of FaceDetection objects.

    Detections are sorted by confidence (descending).  For each detection we
    suppress all remaining detections whose IoU exceeds *iou_threshold*.

    This is a pure-Python implementation to avoid pulling in ``torch`` or
    ``torchvision`` just for NMS.

    Args:
        detections: Unsorted list of detections.
        iou_threshold: Suppress detections with IoU above this value.

    Returns:
        Filtered list of detections (highest confidence kept).
    """
    if len(detections) <= 1:
        return detections

    # Sort descending by confidence so we keep the strongest first
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[FaceDetection] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)

        # Remove detections that overlap too much with `best`
        remaining: List[FaceDetection] = []
        for det in sorted_dets:
            if compute_iou(best.bbox, det.bbox) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep


def apply_padding(detections: List[FaceDetection],
                  pad_ratio: float,
                  image_shape: Tuple[int, ...]) -> List[FaceDetection]:
    """Expand each bounding box by *pad_ratio* on every side.

    Padding ensures that blurring fully covers hairlines and jaw edges that
    tight detections frequently clip.

    Args:
        detections: Detections to pad.
        pad_ratio: Fractional expansion relative to box dimensions.
                   0.15 = expand 15 % on each side.
        image_shape: (H, W, ...) used to clamp boxes inside the frame.

    Returns:
        New list with padded (and clamped) bounding boxes.
    """
    if pad_ratio <= 0.0:
        return detections

    h_img, w_img = image_shape[:2]
    padded: List[FaceDetection] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * pad_ratio)
        dy = int(bh * pad_ratio)

        new_bbox = clip_bbox(
            (x1 - dx, y1 - dy, x2 + dx, y2 + dy),
            w_img, h_img,
        )
        padded.append(FaceDetection(
            bbox=new_bbox,
            confidence=det.confidence,
            landmarks=det.landmarks,
            track_id=det.track_id,
        ))

    return padded


def clip_bbox(bbox: Tuple[int, int, int, int],
              img_width: int,
              img_height: int) -> Tuple[int, int, int, int]:
    """Clamp bounding-box coordinates to stay within image bounds.

    Args:
        bbox: (x1, y1, x2, y2).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Clamped (x1, y1, x2, y2).
    """
    x1 = max(0, min(bbox[0], img_width - 1))
    y1 = max(0, min(bbox[1], img_height - 1))
    x2 = max(0, min(bbox[2], img_width))
    y2 = max(0, min(bbox[3], img_height))
    return (x1, y1, x2, y2)


def filter_by_size(detections: List[FaceDetection],
                   min_size: int = 20,
                   max_size: int = 0) -> List[FaceDetection]:
    """Remove detections whose bounding box is too small or too large.

    Small faces are frequently false positives (texture noise).
    Very large "faces" are typically false positives from wall posters or
    other non-face objects that accidentally trigger the detector.

    Args:
        detections: Input detections.
        min_size: Minimum of (width, height) in pixels.  Detections below
                  this are discarded.
        max_size: Maximum of (width, height).  0 = no upper limit.

    Returns:
        Filtered list.
    """
    filtered: List[FaceDetection] = []
    for det in detections:
        w = det.width
        h = det.height
        face_size = min(w, h)
        face_max = max(w, h)

        if face_size < min_size:
            logger.debug(
                "Dropping small detection: %dx%d < min_size=%d",
                w, h, min_size,
            )
            continue
        if max_size > 0 and face_max > max_size:
            logger.debug(
                "Dropping large detection: %dx%d > max_size=%d",
                w, h, max_size,
            )
            continue
        filtered.append(det)

    return filtered


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class FaceDetector(ABC):
    """Abstract base class for all face detection backends.

    Subclasses implement ``_detect_raw`` to produce raw detections from their
    native backend.  The public ``detect`` method calls ``_detect_raw`` and
    then applies uniform post-processing (NMS, size filter, padding).

    Args:
        config: Pipeline configuration controlling thresholds, padding, etc.
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        self.config = config
        self._name: str = self.__class__.__name__
        logger.info("Initializing detector: %s", self._name)

    # ------ Public API -----------------------------------------------------

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in *image* and apply full post-processing.

        This is the only method downstream code should call.

        Args:
            image: BGR image as HWC numpy array (OpenCV convention).

        Returns:
            Post-processed list of ``FaceDetection`` objects.
        """
        if image is None or image.size == 0:
            logger.warning("%s: received empty/None image — returning []", self._name)
            return []

        try:
            raw = self._detect_raw(image)
        except Exception:
            logger.exception(
                "%s: detection failed on frame (shape=%s)",
                self._name, image.shape,
            )
            return []

        return self._post_process(raw, image.shape)

    # ------ Hook for subclasses --------------------------------------------

    @abstractmethod
    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Backend-specific detection — must be overridden.

        Returns raw detections *before* NMS / size filtering / padding.
        Confidence filtering at ``config.confidence_threshold`` should be
        applied here (or by the native backend) to avoid wasting work in
        post-processing.
        """
        ...

    # ------ Shared post-processing -----------------------------------------

    def _post_process(self, detections: List[FaceDetection],
                      image_shape: Tuple[int, ...]) -> List[FaceDetection]:
        """Apply NMS → size filter → padding → clip.

        Called automatically by ``detect()``.  Subclasses normally do not need
        to override this.

        Args:
            detections: Raw detections from ``_detect_raw``.
            image_shape: (H, W, C) of the original image.

        Returns:
            Cleaned, padded, and clamped detections.
        """
        if not detections:
            return []

        # 1. Non-maximum suppression
        detections = apply_nms(detections, self.config.nms_threshold)

        # 2. Size filter
        detections = filter_by_size(
            detections,
            min_size=self.config.min_face_size,
            max_size=self.config.max_face_size,
        )

        # 3. Padding (expands boxes, then clips to image bounds internally)
        detections = apply_padding(
            detections,
            pad_ratio=self.config.pad_ratio,
            image_shape=image_shape,
        )

        logger.debug(
            "%s: post-process → %d detections", self._name, len(detections),
        )
        return detections


# ---------------------------------------------------------------------------
# SCRFD Detector  (InsightFace, primary backend)
# ---------------------------------------------------------------------------

class SCRFDDetector(FaceDetector):
    """Face detector using InsightFace's SCRFD model (Sample and Computation
    Redistribution for Efficient Face Detection).

    SCRFD achieves 93.7 % AP on the WIDER Face Hard set while running at
    ≈10 ms on a desktop GPU — making it the best accuracy/speed trade-off
    for production privacy blurring.

    The ``buffalo_sc`` model pack bundles a small SCRFD model suitable for
    real-time work; ``buffalo_l`` uses a larger backbone for maximum recall.

    Requires:
        - ``insightface``  (pip install insightface)
        - ``onnxruntime`` or ``onnxruntime-gpu``
    """

    # Model pack names in preference order.  ``buffalo_sc`` is lighter;
    # ``buffalo_l`` is more accurate but heavier.
    _MODEL_PACKS = ("buffalo_sc", "buffalo_l")

    def __init__(self, config: FaceBlurConfig,
                 model_pack: Optional[str] = None) -> None:
        super().__init__(config)

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is required for SCRFDDetector but is not installed.\n"
                "Install it with:\n"
                "    pip install insightface onnxruntime   # CPU\n"
                "    pip install insightface onnxruntime-gpu  # NVIDIA GPU\n"
                "See https://github.com/deepinsight/insightface for details."
            ) from exc

        # Resolve model directory (absolute or relative to project root)
        model_dir = self._resolve_model_dir(config.model_dir)

        # Choose execution providers based on config.device
        providers = self._get_providers(config.device)

        # Try each model pack until one succeeds
        packs = [model_pack] if model_pack else list(self._MODEL_PACKS)
        self._app: Optional[object] = None

        for pack in packs:
            try:
                logger.info(
                    "Loading InsightFace model pack '%s' from %s", pack, model_dir,
                )
                app = FaceAnalysis(
                    name=pack,
                    root=model_dir,
                    providers=providers,
                )
                det_size = config.input_size
                app.prepare(ctx_id=0, det_size=det_size)
                self._app = app
                logger.info(
                    "SCRFDDetector ready — pack=%s, det_size=%s, providers=%s",
                    pack, det_size, providers,
                )
                break
            except Exception:
                logger.warning(
                    "Failed to load model pack '%s', trying next…", pack,
                    exc_info=True,
                )

        if self._app is None:
            raise RuntimeError(
                f"SCRFDDetector: could not load any model pack from {packs}.  "
                f"Make sure insightface model files are available in '{model_dir}'."
            )

    # ------ Raw detection --------------------------------------------------

    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Run InsightFace detection and map results to FaceDetection."""
        faces = self._app.get(image)  # type: ignore[union-attr]
        detections: List[FaceDetection] = []

        for face in faces:
            conf = float(face.det_score)
            if conf < self.config.confidence_threshold:
                continue

            # InsightFace returns bbox as float [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()

            # 5-point landmarks (left_eye, right_eye, nose, mouth_l, mouth_r)
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = np.array(face.kps, dtype=np.float32).reshape(5, 2)

            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                landmarks=landmarks,
            ))

        return detections

    # ------ Helpers --------------------------------------------------------

    @staticmethod
    def _resolve_model_dir(model_dir: str) -> str:
        """Ensure model directory exists and return its absolute path."""
        path = Path(model_dir)
        if not path.is_absolute():
            # Relative to project root (two levels up from this file)
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / model_dir
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def _get_providers(device: str) -> List[str]:
        """Map config.device string to ONNX Runtime execution providers."""
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# RetinaFace Detector  (InsightFace alternative)
# ---------------------------------------------------------------------------

class RetinaFaceDetector(FaceDetector):
    """Face detector using InsightFace's RetinaFace model.

    RetinaFace was the gold-standard single-stage face detector before SCRFD.
    It is slightly heavier but still very accurate.  This implementation
    tries to load ``retinaface_r50_v1`` and falls back to SCRFD model packs
    if the RetinaFace-specific weights are not present.

    Requires:
        - ``insightface``
        - ``onnxruntime`` or ``onnxruntime-gpu``
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        super().__init__(config)

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is required for RetinaFaceDetector.\n"
                "Install with: pip install insightface onnxruntime"
            ) from exc

        model_dir = SCRFDDetector._resolve_model_dir(config.model_dir)
        providers = SCRFDDetector._get_providers(config.device)

        # Preference order: RetinaFace packs → fall back to SCRFD packs
        packs = ["retinaface_r50_v1", "buffalo_l", "buffalo_sc"]
        self._app: Optional[object] = None

        for pack in packs:
            try:
                logger.info(
                    "RetinaFaceDetector: trying model pack '%s'", pack,
                )
                app = FaceAnalysis(
                    name=pack,
                    root=model_dir,
                    providers=providers,
                )
                app.prepare(ctx_id=0, det_size=config.input_size)
                self._app = app
                logger.info(
                    "RetinaFaceDetector ready — pack=%s, providers=%s",
                    pack, providers,
                )
                break
            except Exception:
                logger.warning(
                    "Pack '%s' unavailable, trying next…", pack, exc_info=True,
                )

        if self._app is None:
            raise RuntimeError(
                "RetinaFaceDetector: could not load any model pack. "
                "Run `pip install insightface` and ensure model weights are "
                f"downloaded to '{model_dir}'."
            )

    # ------ Raw detection (same mapping as SCRFD) --------------------------

    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Run InsightFace detection, map to FaceDetection."""
        faces = self._app.get(image)  # type: ignore[union-attr]
        detections: List[FaceDetection] = []

        for face in faces:
            conf = float(face.det_score)
            if conf < self.config.confidence_threshold:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int).tolist()

            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = np.array(face.kps, dtype=np.float32).reshape(5, 2)

            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                landmarks=landmarks,
            ))

        return detections


# ---------------------------------------------------------------------------
# YuNet Detector  (OpenCV built-in — zero external dependency)
# ---------------------------------------------------------------------------

class YuNetDetector(FaceDetector):
    """Face detector using OpenCV's built-in ``FaceDetectorYN`` (YuNet).

    YuNet is a 75 K-parameter CNN that ships with OpenCV ≥ 4.5.4 and requires
    *zero* additional pip packages.  It is the guaranteed fallback when neither
    InsightFace nor MediaPipe are installed.

    The ONNX model file is downloaded automatically from the OpenCV Zoo on
    first use and cached in ``config.model_dir``.

    Reference:
        https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    """

    _MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    _MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

    def __init__(self, config: FaceBlurConfig) -> None:
        super().__init__(config)

        model_path = self._ensure_model(config.model_dir)

        # We create the detector with a *placeholder* input size.  On every
        # frame we call ``setInputSize`` to match the actual frame dimensions
        # so YuNet can handle variable-resolution input.
        w, h = config.input_size
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(w, h),
                score_threshold=config.confidence_threshold,
                nms_threshold=config.nms_threshold,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
        except AttributeError:
            raise RuntimeError(
                "cv2.FaceDetectorYN is not available. "
                "Upgrade OpenCV to ≥ 4.5.4:  pip install opencv-python>=4.5.4"
            )

        logger.info(
            "YuNetDetector ready — model=%s, input_size=(%d,%d)",
            model_path, w, h,
        )

    # ------ Raw detection --------------------------------------------------

    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Run YuNet detection, map results to FaceDetection."""
        h_img, w_img = image.shape[:2]

        # YuNet requires input size to match frame dimensions for best results
        self._detector.setInputSize((w_img, h_img))

        _, raw_faces = self._detector.detect(image)

        if raw_faces is None:
            return []

        detections: List[FaceDetection] = []

        # YuNet output per row: [x, y, w, h, ..., score]
        # Columns 0-3: bbox (x, y, w, h)
        # Columns 4-13: 5 landmark pairs (x0,y0, x1,y1, ..., x4,y4)
        # Column 14: confidence score
        for face in raw_faces:
            conf = float(face[14])
            if conf < self.config.confidence_threshold:
                continue

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            bbox = clip_bbox((x, y, x + w, y + h), w_img, h_img)

            # Extract 5 landmarks from columns 4..13
            landmarks = np.array([
                [face[4],  face[5]],   # right eye
                [face[6],  face[7]],   # left eye
                [face[8],  face[9]],   # nose tip
                [face[10], face[11]],  # right mouth corner
                [face[12], face[13]],  # left mouth corner
            ], dtype=np.float32)

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=conf,
                landmarks=landmarks,
            ))

        return detections

    # ------ Model download -------------------------------------------------

    @classmethod
    def _ensure_model(cls, model_dir: str) -> str:
        """Download the YuNet ONNX model if it is not already cached.

        Args:
            model_dir: Directory to store model files.

        Returns:
            Absolute path to the model file.
        """
        path = Path(model_dir)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / model_dir
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / cls._MODEL_FILENAME

        if model_path.exists():
            logger.debug("YuNet model found at %s", model_path)
            return str(model_path)

        logger.info(
            "Downloading YuNet model from OpenCV Zoo…\n  URL: %s\n  → %s",
            cls._MODEL_URL, model_path,
        )
        try:
            # Download with progress logging
            tmp_path = str(model_path) + ".tmp"
            urllib.request.urlretrieve(cls._MODEL_URL, tmp_path)
            os.replace(tmp_path, str(model_path))
            logger.info("YuNet model downloaded successfully (%s)", model_path)
        except Exception as exc:
            # Clean up partial download
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"Failed to download YuNet model from {cls._MODEL_URL}. "
                f"You can download it manually and place it at {model_path}.\n"
                f"Error: {exc}"
            ) from exc

        return str(model_path)


# ---------------------------------------------------------------------------
# MediaPipe Detector  (optional, ultra-fast CPU)
# ---------------------------------------------------------------------------

class MediaPipeDetector(FaceDetector):
    """Face detector using Google MediaPipe's BlazeFace model.

    BlazeFace is designed for real-time mobile inference and achieves sub-
    millisecond detection on modern CPUs.  It returns 6 keypoints per face
    but only the first 5 are mapped to the standard landmark layout.

    MediaPipe uses *relative* coordinates (0–1).  This class converts them
    to absolute pixel coordinates.

    Requires:
        - ``mediapipe``  (pip install mediapipe)
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        super().__init__(config)

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for MediaPipeDetector but is not installed.\n"
                "Install with:  pip install mediapipe"
            ) from exc

        self._mp_face_detection = mp.solutions.face_detection

        # model_selection: 0 = short-range (< 2 m), 1 = full-range (< 5 m)
        # For privacy blurring we want full-range to catch distant faces.
        self._face_detection = self._mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=config.confidence_threshold,
        )

        logger.info(
            "MediaPipeDetector ready — model=full-range, "
            "min_conf=%.2f", config.confidence_threshold,
        )

    def __del__(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_face_detection") and self._face_detection:
            self._face_detection.close()

    # ------ Raw detection --------------------------------------------------

    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Run MediaPipe detection, convert relative → absolute coords."""
        h_img, w_img = image.shape[:2]

        # MediaPipe expects RGB input
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)

        if not results.detections:
            return []

        detections: List[FaceDetection] = []

        for det in results.detections:
            # MediaPipe confidence
            conf = det.score[0]
            if conf < self.config.confidence_threshold:
                continue

            # Relative bounding box → absolute pixels
            rel_bb = det.location_data.relative_bounding_box
            x1 = int(rel_bb.xmin * w_img)
            y1 = int(rel_bb.ymin * h_img)
            x2 = int((rel_bb.xmin + rel_bb.width) * w_img)
            y2 = int((rel_bb.ymin + rel_bb.height) * h_img)
            bbox = clip_bbox((x1, y1, x2, y2), w_img, h_img)

            # Extract available keypoints (up to 5) as landmarks
            landmarks = None
            if det.location_data.relative_keypoints:
                kps = det.location_data.relative_keypoints
                # MediaPipe order: right_eye(0), left_eye(1), nose_tip(2),
                # mouth_center(3), right_ear(4), left_ear(5)
                # Map to InsightFace order:
                # left_eye, right_eye, nose, left_mouth, right_mouth
                # We approximate mouth corners from the mouth_center keypoint
                landmark_list = []
                for kp in kps[:5]:
                    landmark_list.append([kp.x * w_img, kp.y * h_img])
                # Pad to 5 if fewer keypoints available
                while len(landmark_list) < 5:
                    landmark_list.append(landmark_list[-1])
                landmarks = np.array(landmark_list[:5], dtype=np.float32)

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=conf,
                landmarks=landmarks,
            ))

        return detections


# ---------------------------------------------------------------------------
# Ensemble Detector  (multi-backend fusion)
# ---------------------------------------------------------------------------

class EnsembleDetector(FaceDetector):
    """Runs multiple detector backends and merges results with cross-detector NMS.

    This maximizes recall — if *any* detector finds a face, the ensemble will
    include it in the output.  Overlapping detections from different backends
    are merged by keeping the highest-confidence instance.

    The ensemble is constructed from a list of detector name strings.  Each
    name is resolved by ``create_detector`` (which itself handles fallbacks).

    Example config:
        config.detector = "ensemble"
        # The ensemble will automatically try: scrfd → yunet → mediapipe
    """

    # Default set of detectors to include in the ensemble
    _DEFAULT_BACKENDS = ["scrfd", "yunet"]

    def __init__(self, config: FaceBlurConfig,
                 backend_names: Optional[List[str]] = None) -> None:
        super().__init__(config)

        if backend_names is None:
            backend_names = list(self._DEFAULT_BACKENDS)

        self._detectors: List[FaceDetector] = []
        self._detector_names: List[str] = []

        for name in backend_names:
            try:
                # Use a temporary config with the backend name to create each
                # sub-detector.  We avoid infinite recursion because the
                # factory won't recurse into EnsembleDetector for non-"ensemble"
                # names.
                sub_config = FaceBlurConfig(
                    detector=name,
                    confidence_threshold=config.confidence_threshold,
                    nms_threshold=config.nms_threshold,
                    min_face_size=config.min_face_size,
                    max_face_size=config.max_face_size,
                    pad_ratio=0.0,  # Padding applied once at ensemble level
                    input_size=config.input_size,
                    device=config.device,
                    model_dir=config.model_dir,
                )
                det = _create_single_detector(name, sub_config)
                self._detectors.append(det)
                self._detector_names.append(name)
                logger.info("Ensemble: added backend '%s'", name)
            except Exception:
                logger.warning(
                    "Ensemble: backend '%s' unavailable — skipping", name,
                    exc_info=True,
                )

        if not self._detectors:
            raise RuntimeError(
                "EnsembleDetector: no backends could be loaded from "
                f"{backend_names}. At least one detector must be available."
            )

        logger.info(
            "EnsembleDetector ready — %d backends: %s",
            len(self._detectors), self._detector_names,
        )

    # ------ Raw detection --------------------------------------------------

    def _detect_raw(self, image: np.ndarray) -> List[FaceDetection]:
        """Run every backend and merge all detections.

        Cross-detector NMS is applied in ``_post_process`` (inherited).
        """
        all_detections: List[FaceDetection] = []

        for det, name in zip(self._detectors, self._detector_names):
            try:
                results = det._detect_raw(image)
                logger.debug(
                    "Ensemble: '%s' returned %d raw detections",
                    name, len(results),
                )
                all_detections.extend(results)
            except Exception:
                logger.warning(
                    "Ensemble: '%s' failed on this frame — skipping",
                    name, exc_info=True,
                )

        return all_detections


# ---------------------------------------------------------------------------
# Internal single-detector factory (avoids circular logic with ensemble)
# ---------------------------------------------------------------------------

# Registry mapping config string → detector class
_DETECTOR_REGISTRY: Dict[str, type] = {
    "scrfd": SCRFDDetector,
    "retinaface": RetinaFaceDetector,
    "yunet": YuNetDetector,
    "mediapipe": MediaPipeDetector,
}


def _create_single_detector(name: str,
                             config: FaceBlurConfig) -> FaceDetector:
    """Instantiate a single (non-ensemble) detector by name.

    This is used internally by both ``create_detector`` and
    ``EnsembleDetector`` to avoid infinite recursion.

    Args:
        name: Detector backend name (must be in ``_DETECTOR_REGISTRY``).
        config: Pipeline configuration.

    Returns:
        Initialized ``FaceDetector`` subclass.

    Raises:
        ValueError: If *name* is not recognized.
        RuntimeError / ImportError: If the backend cannot be loaded.
    """
    cls = _DETECTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{name}'. "
            f"Valid options: {sorted(_DETECTOR_REGISTRY.keys())}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# Public Factory
# ---------------------------------------------------------------------------

def create_detector(config: FaceBlurConfig) -> FaceDetector:
    """Factory function — create the appropriate face detector.

    Implements a graceful fallback chain:

    1. Try the detector requested in ``config.detector``.
    2. If that fails (missing package, bad model), try SCRFD.
    3. If SCRFD fails, try YuNet (zero-dependency fallback).
    4. If everything fails, raise ``RuntimeError``.

    Args:
        config: Pipeline configuration.

    Returns:
        Ready-to-use ``FaceDetector`` instance.

    Raises:
        RuntimeError: If no detector backend could be loaded at all.
    """
    requested = config.detector.lower()
    logger.info("Requested detector: '%s'", requested)

    # --- Ensemble: special handling ----------------------------------------
    if requested == "ensemble":
        try:
            return EnsembleDetector(config)
        except Exception:
            logger.warning(
                "EnsembleDetector failed to initialize — falling back to "
                "single detector",
                exc_info=True,
            )
            # Continue to single-detector fallback chain below
            requested = "scrfd"

    # --- Single detector with fallback chain --------------------------------
    fallback_chain = _build_fallback_chain(requested)

    for name in fallback_chain:
        try:
            detector = _create_single_detector(name, config)
            if name != requested:
                logger.warning(
                    "Primary detector '%s' unavailable — fell back to '%s'",
                    requested, name,
                )
            else:
                logger.info("Detector '%s' loaded successfully", name)
            return detector
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "Detector '%s' failed to load: %s", name, exc,
            )
        except Exception:
            logger.warning(
                "Detector '%s' failed unexpectedly", name, exc_info=True,
            )

    # If we get here, nothing worked
    raise RuntimeError(
        "FATAL: No face detection backend could be loaded.\n"
        "Tried: " + " → ".join(fallback_chain) + "\n"
        "Install at least one:\n"
        "  pip install insightface onnxruntime   # For SCRFD/RetinaFace\n"
        "  pip install opencv-python>=4.5.4      # For YuNet (built-in)\n"
        "  pip install mediapipe                  # For MediaPipe BlazeFace"
    )


def _build_fallback_chain(requested: str) -> List[str]:
    """Build an ordered fallback sequence starting from *requested*.

    The chain always ends with ``yunet`` because it has zero extra
    dependencies beyond OpenCV itself.

    Args:
        requested: User-requested detector name.

    Returns:
        Ordered list of detector names to try.
    """
    # Canonical fallback order
    full_order = ["scrfd", "retinaface", "yunet", "mediapipe"]

    chain: List[str] = [requested]
    for name in full_order:
        if name not in chain:
            chain.append(name)

    return chain
