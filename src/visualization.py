"""
Visualisation Module — Debug Overlays for Face Blur Pipeline
==============================================================

Provides debug-mode drawing routines for bounding boxes, landmarks,
track IDs, and a real-time HUD (heads-up display) with performance stats.

These overlays are ONLY used when `config.debug_mode = True`. Production
output must never include annotations — the blur module handles that.

Design Principles:
    - Semi-transparent overlays for professional appearance
    - Hash-based per-track colour generation for consistent identity colours
    - All functions return NEW images (never mutate the input)
    - Efficient numpy operations — overlays add < 1ms per frame on CPU
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import FaceBlurConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FaceDetection (local copy to avoid circular imports)
# ---------------------------------------------------------------------------

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
# Colour utilities
# ---------------------------------------------------------------------------

# Hand-picked palette of 20 visually distinct colours (BGR format).
# Used as the primary palette; for track IDs beyond 20, we hash.
_PALETTE = [
    (255, 76, 76),     # coral red
    (76, 255, 76),     # lime green
    (76, 76, 255),     # blue
    (255, 255, 76),    # yellow
    (255, 76, 255),    # magenta
    (76, 255, 255),    # cyan
    (255, 165, 76),    # orange
    (165, 76, 255),    # purple
    (76, 255, 165),    # spring green
    (255, 76, 165),    # pink
    (76, 165, 255),    # sky blue
    (200, 200, 76),    # olive
    (200, 76, 200),    # plum
    (76, 200, 200),    # teal
    (255, 200, 150),   # peach
    (150, 200, 255),   # light blue
    (200, 255, 150),   # light green
    (255, 150, 200),   # light pink
    (150, 255, 200),   # mint
    (200, 150, 255),   # lavender
]


def _get_track_colour(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent BGR colour for a given track ID.

    Uses the palette for small IDs and a deterministic hash for larger ones,
    ensuring the same track ID always maps to the same colour across frames.

    Args:
        track_id: Track identity (≥ 0). Untracked (-1) defaults to white.

    Returns:
        BGR colour tuple.
    """
    if track_id < 0:
        return (255, 255, 255)  # white for untracked detections

    if track_id < len(_PALETTE):
        return _PALETTE[track_id % len(_PALETTE)]

    # Deterministic hash → hue-based colour for IDs beyond the palette.
    # We use a golden-ratio-based hash to maximise colour spread.
    hue = int((track_id * 37) % 180)  # OpenCV hue range is 0–179
    hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def _overlay_transparent(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend an overlay onto a base image with transparency.

    This is used for semi-transparent HUD backgrounds and highlight
    rectangles. Uses the standard alpha compositing formula:
        output = alpha * overlay + (1 - alpha) * base

    Args:
        base: Background image (modified in-place for performance).
        overlay: Foreground image (same shape as base).
        alpha: Transparency factor (0 = invisible, 1 = opaque).

    Returns:
        Blended image (same reference as `base` — modified in-place).
    """
    cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0, dst=base)
    return base


# ---------------------------------------------------------------------------
# Public API — drawing functions
# ---------------------------------------------------------------------------

def draw_detections(
    image: np.ndarray,
    detections: List[FaceDetection],
    config: FaceBlurConfig,
) -> np.ndarray:
    """Draw bounding boxes, confidence scores, and track IDs on an image.

    Each detection gets:
        - A coloured bounding box (colour keyed to track_id)
        - A label showing "ID: <track_id>  <confidence%>"
        - A semi-transparent label background for readability

    Args:
        image: Input BGR image (not modified).
        detections: Face detections to visualise.
        config: Pipeline configuration.

    Returns:
        New image with detection overlays drawn.
    """
    output = image.copy()

    if not detections:
        return output

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        colour = _get_track_colour(det.track_id)
        thickness = 2

        # ── Bounding box ──
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, thickness)

        # ── Label text ──
        if det.track_id >= 0:
            label = f"ID:{det.track_id} {det.confidence:.0%}"
        else:
            label = f"{det.confidence:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Label background (semi-transparent)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = y1
        label_x2 = min(output.shape[1], x1 + text_w + 8)

        # Draw semi-transparent background rectangle
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (x1, label_y1),
            (label_x2, label_y2),
            colour,
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, dst=output)

        # Draw label text
        text_colour = (0, 0, 0)  # black text on coloured background
        cv2.putText(
            output,
            label,
            (x1 + 4, y1 - baseline - 2),
            font,
            font_scale,
            text_colour,
            font_thickness,
            cv2.LINE_AA,
        )

        # ── Confidence bar (thin line along bottom of bbox) ──
        bar_length = int((x2 - x1) * det.confidence)
        cv2.line(
            output,
            (x1, y2 + 3),
            (x1 + bar_length, y2 + 3),
            colour,
            2,
        )

    return output


def draw_landmarks(
    image: np.ndarray,
    detections: List[FaceDetection],
) -> np.ndarray:
    """Draw 5-point facial landmarks on an image.

    Landmarks are drawn as small coloured circles:
        - Left eye  (0): green
        - Right eye (1): green
        - Nose      (2): blue
        - Left mouth corner  (3): red
        - Right mouth corner (4): red

    Args:
        image: Input BGR image (not modified).
        detections: Face detections (only those with landmarks are drawn).

    Returns:
        New image with landmark overlays.
    """
    output = image.copy()

    # Landmark colours (BGR): eyes=green, nose=blue, mouth=red
    landmark_colours = [
        (0, 255, 0),    # left eye
        (0, 255, 0),    # right eye
        (255, 128, 0),  # nose
        (0, 0, 255),    # left mouth
        (0, 0, 255),    # right mouth
    ]

    for det in detections:
        if det.landmarks is None:
            continue

        landmarks = det.landmarks
        if landmarks.ndim != 2 or landmarks.shape[0] < 5:
            logger.warning(
                "Unexpected landmark shape %s for track %d — skipping",
                landmarks.shape, det.track_id,
            )
            continue

        for i in range(min(5, landmarks.shape[0])):
            x = int(round(landmarks[i, 0]))
            y = int(round(landmarks[i, 1]))
            colour = landmark_colours[i] if i < len(landmark_colours) else (255, 255, 255)

            # Outer ring (larger, semi-transparent)
            cv2.circle(output, (x, y), 4, colour, 1, cv2.LINE_AA)
            # Inner dot (solid)
            cv2.circle(output, (x, y), 2, colour, cv2.FILLED, cv2.LINE_AA)

    return output


def draw_info_overlay(
    image: np.ndarray,
    fps: float,
    frame_num: int,
    num_faces: int,
) -> np.ndarray:
    """Draw a heads-up display (HUD) with real-time pipeline statistics.

    The HUD is placed in the top-left corner with a semi-transparent
    dark background for readability on any content.

    Displays:
        - Current FPS
        - Frame number
        - Number of detected faces

    Args:
        image: Input BGR image (not modified).
        fps: Current processing frames per second.
        frame_num: Current frame number (1-indexed).
        num_faces: Number of faces detected / tracked in this frame.

    Returns:
        New image with HUD overlay.
    """
    output = image.copy()

    # ── HUD text lines ──
    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_num}",
        f"Faces: {num_faces}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 28
    padding = 10

    # Calculate HUD dimensions
    max_text_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, tw)

    hud_w = max_text_width + padding * 2
    hud_h = line_height * len(lines) + padding * 2

    # ── Semi-transparent dark background ──
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (hud_w, hud_h),
        (30, 30, 30),  # near-black
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, dst=output)

    # ── Draw border ──
    cv2.rectangle(output, (0, 0), (hud_w, hud_h), (100, 100, 100), 1)

    # ── Draw text lines ──
    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_height - 6

        # Colour code: green FPS if > 20, yellow if 10-20, red if < 10
        if i == 0:  # FPS line
            if fps >= 20:
                text_colour = (0, 230, 0)    # green
            elif fps >= 10:
                text_colour = (0, 230, 230)  # yellow
            else:
                text_colour = (0, 0, 230)    # red
        elif i == 2:  # Faces line
            text_colour = (230, 180, 50)     # light blue
        else:
            text_colour = (220, 220, 220)    # light grey

        cv2.putText(
            output,
            line,
            (padding, y),
            font,
            font_scale,
            text_colour,
            font_thickness,
            cv2.LINE_AA,
        )

    return output


# ---------------------------------------------------------------------------
# Composite debug frame
# ---------------------------------------------------------------------------

def draw_debug_frame(
    image: np.ndarray,
    detections: List[FaceDetection],
    config: FaceBlurConfig,
    fps: float = 0.0,
    frame_num: int = 0,
) -> np.ndarray:
    """Draw all debug overlays on a single frame.

    Convenience function that composites bounding boxes, landmarks,
    and the HUD in a single call. Used by the video pipeline when
    `config.debug_mode = True`.

    Args:
        image: Input BGR image (not modified).
        detections: Face detections with optional landmarks and track IDs.
        config: Pipeline configuration.
        fps: Current processing FPS.
        frame_num: Current frame number.

    Returns:
        New image with all debug overlays.
    """
    output = draw_detections(image, detections, config)
    output = draw_landmarks(output, detections)
    output = draw_info_overlay(output, fps, frame_num, len(detections))
    return output
