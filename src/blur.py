"""
Blur Engine Module — Face Anonymisation Strategies
====================================================

Provides pluggable blur backends for face anonymisation. Each engine
implements the same interface so they can be swapped via config.

Engines:
    GaussianBlurEngine  — Smooth, natural-looking privacy blur
    PixelateBlurEngine  — Blocky mosaic effect, harder to reverse
    AdaptiveBlurEngine  — Blur strength proportional to detection confidence

Engineering Decisions:
    - All engines work on a COPY of the input image — the original is never
      mutated, which is critical when the same frame is used for both debug
      visualisation and production output.
    - Elliptical masks are generated per-face to follow the natural oval shape
      of a human face, avoiding "blocky" rectangular blur artefacts.
    - Feathering uses a Gaussian-blurred alpha mask to smoothly blend the
      blur boundary into the surrounding image (avoids harsh edges).
    - Boundary clipping is handled explicitly — faces at image edges have
      their ROI clamped to valid pixel coordinates.
    - All kernel sizes are forced to be odd and ≥ 3 (OpenCV requirement).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import FaceBlurConfig

logger = logging.getLogger(__name__)

# Re-declare FaceDetection locally to avoid circular imports.
# Structurally identical to the canonical definition in detector.py.

@dataclass
class FaceDetection:
    """Single face detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence in [0, 1].
        landmarks: Optional 5×2 array of facial landmarks.
        track_id: Assigned tracker identity (-1 = untracked).
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    track_id: int = -1


# ---------------------------------------------------------------------------
# Helpers — shared across all blur engines
# ---------------------------------------------------------------------------

def _ensure_odd(value: int, minimum: int = 3) -> int:
    """Ensure a kernel size is odd and at least `minimum`.

    OpenCV's GaussianBlur requires odd kernel dimensions. This is called
    on every blur operation to guarantee validity.
    """
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value


def _clip_roi(
    bbox: Tuple[int, int, int, int],
    img_h: int,
    img_w: int,
) -> Tuple[int, int, int, int]:
    """Clip a bounding box to valid image coordinates.

    Faces at image borders can have coordinates outside [0, W) × [0, H).
    This clamps them and returns the valid sub-region.

    Returns:
        (x1, y1, x2, y2) clipped to image bounds.
        If the clipped region is degenerate (zero area), returns (-1,-1,-1,-1).
    """
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(img_w, int(bbox[2]))
    y2 = min(img_h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return (-1, -1, -1, -1)
    return (x1, y1, x2, y2)


def _create_elliptical_mask(height: int, width: int) -> np.ndarray:
    """Generate a single-channel elliptical mask for face-shaped blur.

    The ellipse is centred in the ROI and sized to fill it, matching
    the natural oval shape of a human face.

    Args:
        height: ROI height in pixels.
        width: ROI width in pixels.

    Returns:
        (height, width) float32 mask with values in [0, 1].
    """
    mask = np.zeros((height, width), dtype=np.float32)
    centre = (width // 2, height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    return mask


def _feather_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Apply Gaussian feathering to a binary/float mask.

    Blurs the mask edges so the transition from blurred to un-blurred
    regions is gradual, producing a professional appearance.

    Args:
        mask: (H, W) float32 mask in [0, 1].
        radius: Feathering radius in pixels.

    Returns:
        Feathered mask with smooth edges.
    """
    ksize = _ensure_odd(radius * 2 + 1, minimum=3)
    feathered = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    # Re-normalise to [0, 1] in case blur pushed values outside range
    max_val = feathered.max()
    if max_val > 0:
        feathered /= max_val
    return feathered


def _apply_masked_blur(
    image: np.ndarray,
    blurred_roi: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    use_elliptical: bool,
    feather: bool,
    feather_radius: int,
) -> None:
    """Composite a blurred ROI back into the image using an optional mask.

    Modifies `image` in-place within the ROI region.

    Args:
        image: Full output image (modified in-place).
        blurred_roi: The blurred face region, same size as (y2-y1, x2-x1).
        x1, y1, x2, y2: Clipped ROI coordinates.
        use_elliptical: Whether to apply an elliptical mask.
        feather: Whether to feather mask edges.
        feather_radius: Pixel radius for feathering.
    """
    roi_h = y2 - y1
    roi_w = x2 - x1

    if not use_elliptical:
        # Simple rectangular replacement — just paste the blurred region
        image[y1:y2, x1:x2] = blurred_roi
        return

    # Build mask
    mask = _create_elliptical_mask(roi_h, roi_w)

    if feather and feather_radius > 0:
        mask = _feather_mask(mask, feather_radius)

    # Expand mask to 3 channels for alpha blending
    mask_3ch = mask[:, :, np.newaxis]

    # Alpha blend: output = blurred * mask + original * (1 - mask)
    original_roi = image[y1:y2, x1:x2].astype(np.float32)
    blurred_f = blurred_roi.astype(np.float32)
    blended = blurred_f * mask_3ch + original_roi * (1.0 - mask_3ch)
    image[y1:y2, x1:x2] = blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BlurEngine(ABC):
    """Abstract base class for face blur engines.

    All engines must implement `apply()`, which takes an image and a list
    of face detections and returns the image with faces blurred.
    """

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        config: FaceBlurConfig,
    ) -> np.ndarray:
        """Apply face blur to an image.

        Args:
            image: Input image in BGR format (H, W, 3), uint8.
            detections: Face detections with bounding boxes.
            config: Pipeline configuration.

        Returns:
            New image with faces blurred. The original is NOT modified.
        """
        ...


# ---------------------------------------------------------------------------
# Gaussian Blur Engine
# ---------------------------------------------------------------------------

class GaussianBlurEngine(BlurEngine):
    """Smooth Gaussian blur for natural-looking face anonymisation.

    This is the default engine. Gaussian blur is the industry standard
    for privacy blur because it is smooth, fast, and very difficult to
    reverse (unlike simple pixelation which can sometimes be undone with
    super-resolution models).

    Supports:
        - Elliptical mask for face-shaped blur
        - Edge feathering for smooth transitions
        - Configurable blur strength (kernel size)
    """

    def apply(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        config: FaceBlurConfig,
    ) -> np.ndarray:
        """Apply Gaussian blur to all detected faces.

        Args:
            image: Input BGR image.
            detections: Face detections.
            config: Pipeline configuration.

        Returns:
            Copy of image with Gaussian-blurred face regions.
        """
        if not detections:
            return image.copy()

        output = image.copy()
        img_h, img_w = output.shape[:2]
        ksize = _ensure_odd(config.blur_strength)

        for det in detections:
            x1, y1, x2, y2 = _clip_roi(det.bbox, img_h, img_w)
            if x1 < 0:
                # Degenerate ROI — skip this face
                continue

            roi = output[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            _apply_masked_blur(
                output, blurred, x1, y1, x2, y2,
                use_elliptical=config.use_elliptical_mask,
                feather=config.feather_edges,
                feather_radius=config.feather_radius,
            )

        return output


# ---------------------------------------------------------------------------
# Pixelate Blur Engine
# ---------------------------------------------------------------------------

class PixelateBlurEngine(BlurEngine):
    """Mosaic / pixelation blur for face anonymisation.

    Pixelation is visually distinctive and conveys "anonymised" more
    clearly to viewers. It works by downscaling the face region to a
    very small resolution (e.g. 10×10 blocks), then upscaling back with
    nearest-neighbour interpolation, producing the classic blocky look.

    While recent research has shown that pixelation can sometimes be
    reversed by ML models, using very low block counts (5-8) makes
    reversal practically impossible.
    """

    def apply(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        config: FaceBlurConfig,
    ) -> np.ndarray:
        """Apply pixelation blur to all detected faces.

        Args:
            image: Input BGR image.
            detections: Face detections.
            config: Pipeline configuration.

        Returns:
            Copy of image with pixelated face regions.
        """
        if not detections:
            return image.copy()

        output = image.copy()
        img_h, img_w = output.shape[:2]
        blocks = max(2, config.pixelate_blocks)  # minimum 2×2 blocks

        for det in detections:
            x1, y1, x2, y2 = _clip_roi(det.bbox, img_h, img_w)
            if x1 < 0:
                continue

            roi = output[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]

            # Guard against very small ROIs where blocks > dimensions
            small_w = max(1, min(blocks, roi_w))
            small_h = max(1, min(blocks, roi_h))

            # Downscale to tiny resolution
            small = cv2.resize(
                roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR
            )
            # Upscale back with nearest-neighbour for blocky effect
            pixelated = cv2.resize(
                small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
            )

            _apply_masked_blur(
                output, pixelated, x1, y1, x2, y2,
                use_elliptical=config.use_elliptical_mask,
                feather=config.feather_edges,
                feather_radius=config.feather_radius,
            )

        return output


# ---------------------------------------------------------------------------
# Adaptive Blur Engine
# ---------------------------------------------------------------------------

class AdaptiveBlurEngine(BlurEngine):
    """Confidence-adaptive Gaussian blur.

    Blur strength is proportional to detection confidence:
        - High confidence (e.g. 0.95) → full blur_strength
        - Low confidence (e.g. 0.30) → reduced blur (blur_strength * 0.7 * conf)

    This is useful when processing scenes where some detections are
    uncertain (e.g. face-like objects, mannequins). Low-confidence
    detections get a lighter blur, preserving more detail if they
    turn out to be false positives, while certain faces get full
    anonymisation.

    A minimum kernel size of 15 ensures even low-confidence detections
    receive meaningful obscuration.
    """

    # Minimum Gaussian kernel size — ensures some blur even at very low confidence
    _MIN_KERNEL: int = 15
    # Scaling factor applied to confidence for blur reduction
    _CONFIDENCE_SCALE: float = 0.7

    def apply(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        config: FaceBlurConfig,
    ) -> np.ndarray:
        """Apply confidence-adaptive Gaussian blur to all detected faces.

        Args:
            image: Input BGR image.
            detections: Face detections with confidence scores.
            config: Pipeline configuration.

        Returns:
            Copy of image with adaptively blurred face regions.
        """
        if not detections:
            return image.copy()

        output = image.copy()
        img_h, img_w = output.shape[:2]
        base_strength = config.blur_strength

        for det in detections:
            x1, y1, x2, y2 = _clip_roi(det.bbox, img_h, img_w)
            if x1 < 0:
                continue

            # Scale blur strength by confidence
            # High confidence → full strength; low → reduced
            conf = max(0.0, min(1.0, det.confidence))
            adaptive_strength = int(
                base_strength * conf * self._CONFIDENCE_SCALE
            )
            adaptive_strength = max(adaptive_strength, self._MIN_KERNEL)
            ksize = _ensure_odd(adaptive_strength)

            roi = output[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            _apply_masked_blur(
                output, blurred, x1, y1, x2, y2,
                use_elliptical=config.use_elliptical_mask,
                feather=config.feather_edges,
                feather_radius=config.feather_radius,
            )

            logger.debug(
                "Adaptive blur: conf=%.2f → kernel=%d (base=%d)",
                conf, ksize, base_strength,
            )

        return output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENGINE_REGISTRY = {
    "gaussian": GaussianBlurEngine,
    "pixelate": PixelateBlurEngine,
    "adaptive": AdaptiveBlurEngine,
}


def create_blur_engine(config: FaceBlurConfig) -> BlurEngine:
    """Factory function to create the appropriate blur engine.

    Looks up `config.blur_type` in the registry and returns an instance.

    Args:
        config: Pipeline configuration with `blur_type` field.

    Returns:
        Instantiated BlurEngine subclass.

    Raises:
        ValueError: If `blur_type` is not recognised.
    """
    blur_type = config.blur_type.lower()
    engine_cls = _ENGINE_REGISTRY.get(blur_type)

    if engine_cls is None:
        valid = ", ".join(sorted(_ENGINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown blur_type '{config.blur_type}'. "
            f"Valid options: {valid}"
        )

    engine = engine_cls()
    logger.info("Blur engine created: %s", engine_cls.__name__)
    return engine
