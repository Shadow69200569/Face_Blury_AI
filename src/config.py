"""
Configuration Module for Face Blur Pipeline
=============================================

Centralized configuration using Python dataclasses. All tunable parameters
for detection, tracking, blurring, and pipeline behavior are defined here.

This module provides:
    - FaceBlurConfig: Main configuration dataclass with sensible defaults
    - Config validation and auto-correction
    - Device auto-detection (CPU/CUDA)
    - Preset configurations for common use cases

Engineering Decision:
    Using @dataclass instead of YAML/JSON config files because:
    1. Type safety — Python type hints catch errors at IDE level
    2. Default values — Every parameter has a sensible default
    3. Validation — Can add __post_init__ validation logic
    4. CLI integration — Easy to map argparse args to config fields
    5. No external dependency — No pyyaml/toml needed
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations for type-safe configuration
# ---------------------------------------------------------------------------

class DetectorType(Enum):
    """Supported face detection backends."""
    SCRFD = "scrfd"                # InsightFace SCRFD — best accuracy (93.7% WIDER Hard)
    RETINAFACE = "retinaface"      # InsightFace RetinaFace — gold standard
    YUNET = "yunet"                # OpenCV built-in — zero dependency (75K params)
    MEDIAPIPE = "mediapipe"        # Google MediaPipe BlazeFace — ultra-fast CPU
    ENSEMBLE = "ensemble"          # Runs multiple detectors, merges with NMS


class BlurType(Enum):
    """Supported blur methods."""
    GAUSSIAN = "gaussian"          # Smooth, natural-looking privacy blur
    PIXELATE = "pixelate"          # Blocky mosaic — harder to reverse
    ADAPTIVE = "adaptive"          # Blur intensity proportional to confidence


class DeviceType(Enum):
    """Compute device selection."""
    AUTO = "auto"                  # Auto-detect GPU, fall back to CPU
    CPU = "cpu"                    # Force CPU inference
    CUDA = "cuda"                 # Force NVIDIA GPU inference


# ---------------------------------------------------------------------------
# Main Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaceBlurConfig:
    """
    Complete configuration for the face blur pipeline.
    
    All parameters have sensible production defaults. Override via CLI args
    or by constructing with keyword arguments.
    
    Example:
        config = FaceBlurConfig(detector="scrfd", blur_type="pixelate")
        config = FaceBlurConfig.fast_preset()  # Quick processing preset
    """

    # ---- Detection Parameters ----
    detector: str = "scrfd"
    """Face detector backend: 'scrfd', 'retinaface', 'yunet', 'mediapipe', 'ensemble'"""
    
    confidence_threshold: float = 0.5
    """Minimum detection confidence to consider a face. Lower = more sensitive
    but more false positives. Recommended: 0.3-0.5 for privacy (catch everything),
    0.6-0.8 for precision-critical applications."""
    
    nms_threshold: float = 0.4
    """Non-maximum suppression IoU threshold. Controls overlap tolerance
    when merging duplicate detections. Lower = more aggressive merging."""
    
    min_face_size: int = 20
    """Minimum face bounding box size in pixels. Detections smaller than this
    are filtered out to reduce false positives from noise."""
    
    max_face_size: int = 0
    """Maximum face bounding box size in pixels. 0 = no limit.
    Useful to ignore very large false positives."""
    
    pad_ratio: float = 0.15
    """Expand detected bounding box by this ratio on each side.
    Ensures partial faces and hairlines are fully covered by blur.
    0.1-0.2 recommended for privacy applications."""
    
    input_size: Tuple[int, int] = (640, 640)
    """Detection model input resolution. Higher = better accuracy on small faces
    but slower. (640, 640) is the standard for SCRFD/RetinaFace."""

    # ---- Blur Parameters ----
    blur_type: str = "gaussian"
    """Blur method: 'gaussian' (smooth), 'pixelate' (mosaic), 'adaptive' (confidence-based)"""
    
    blur_strength: int = 99
    """Gaussian blur kernel size. Must be odd. Higher = more blur.
    51 = moderate, 99 = strong (default), 151 = very heavy."""
    
    pixelate_blocks: int = 10
    """Number of pixel blocks for pixelation blur. Lower = more pixelated.
    5-10 for strong anonymization, 15-20 for subtle effect."""
    
    use_elliptical_mask: bool = True
    """Apply blur through an elliptical mask for natural face-shaped blur.
    If False, uses rectangular blur region."""
    
    feather_edges: bool = True
    """Blend blur edges smoothly into surrounding image.
    Prevents hard, visible blur boundaries."""
    
    feather_radius: int = 15
    """Pixel radius for edge feathering. Higher = smoother transition."""

    # ---- Video Pipeline Parameters ----
    frame_skip: int = 0
    """Process face detection every N frames. 0 = process every frame.
    2-3 recommended for real-time with tracking enabled.
    Skipped frames use tracker predictions instead of fresh detections."""
    
    batch_size: int = 1
    """Number of frames to batch for GPU inference. >1 only useful with
    GPU and sufficient VRAM. Typical: 1-4."""
    
    preserve_audio: bool = True
    """Mux original audio track into output video via FFmpeg.
    Requires FFmpeg installed and on PATH."""
    
    output_codec: str = "mp4v"
    """Video output codec FourCC code. 'mp4v' for MP4, 'XVID' for AVI.
    'avc1' or 'H264' for H.264 if available."""
    
    output_quality: int = 95
    """JPEG quality for image output (1-100). Also affects video
    encoding quality when applicable."""

    # ---- Tracking Parameters ----
    enable_tracking: bool = True
    """Enable ByteTrack face tracking across video frames.
    Prevents blur flickering and enables frame-skip optimization."""
    
    track_buffer: int = 30
    """Number of frames to keep lost tracks before deletion.
    Higher = more tolerant of temporary occlusions (face behind hand, etc.)."""
    
    match_threshold: float = 0.8
    """IoU threshold for matching detections to existing tracks.
    Lower = more aggressive matching (fewer ID switches but more errors)."""
    
    track_high_thresh: float = 0.5
    """Confidence threshold for first-stage track association.
    Detections above this are matched first."""
    
    track_low_thresh: float = 0.1
    """Confidence threshold for second-stage track association.
    Low-confidence detections are matched to remaining unmatched tracks.
    This is ByteTrack's key innovation — recovers partially visible faces."""

    # ---- Temporal Smoothing ----
    temporal_smoothing: bool = True
    """Smooth bounding box positions across frames using exponential
    moving average. Reduces jittery blur that looks unprofessional."""
    
    smoothing_alpha: float = 0.7
    """EMA smoothing factor. 0.5 = heavy smoothing (laggy),
    0.7 = moderate (default), 0.9 = light smoothing (responsive)."""

    # ---- System Parameters ----
    device: str = "auto"
    """Compute device: 'auto' (detect GPU), 'cpu', 'cuda'."""
    
    num_workers: int = 4
    """Number of parallel workers for batch processing."""
    
    log_level: str = "INFO"
    """Logging verbosity: 'DEBUG', 'INFO', 'WARNING', 'ERROR'."""
    
    model_dir: str = "models"
    """Directory to store downloaded model weights."""
    
    debug_mode: bool = False
    """Enable debug visualization (bounding boxes, landmarks, track IDs).
    Output includes annotations — do NOT use for production blur output."""

    def __post_init__(self):
        """Validate and auto-correct configuration after initialization."""
        # Ensure blur_strength is odd (required by OpenCV GaussianBlur)
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1
            logger.warning(
                f"blur_strength must be odd. Auto-corrected to {self.blur_strength}"
            )
        
        # Clamp confidence threshold
        self.confidence_threshold = max(0.01, min(0.99, self.confidence_threshold))
        
        # Validate detector name
        valid_detectors = {e.value for e in DetectorType}
        if self.detector not in valid_detectors:
            raise ValueError(
                f"Unknown detector '{self.detector}'. "
                f"Valid options: {valid_detectors}"
            )
        
        # Validate blur type
        valid_blur = {e.value for e in BlurType}
        if self.blur_type not in valid_blur:
            raise ValueError(
                f"Unknown blur_type '{self.blur_type}'. "
                f"Valid options: {valid_blur}"
            )
        
        # Auto-detect device
        if self.device == "auto":
            self.device = self._detect_device()
            logger.info(f"Auto-detected device: {self.device}")

    @staticmethod
    def _detect_device() -> str:
        """Detect available compute device."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # ---- Preset Configurations ----
    
    @classmethod
    def fast_preset(cls) -> "FaceBlurConfig":
        """Fast processing preset — sacrifices accuracy for speed.
        Uses YuNet detector, frame skipping, no temporal smoothing."""
        return cls(
            detector="yunet",
            confidence_threshold=0.6,
            frame_skip=2,
            temporal_smoothing=False,
            blur_type="gaussian",
            blur_strength=51,
            device="cpu",
        )
    
    @classmethod
    def accurate_preset(cls) -> "FaceBlurConfig":
        """Maximum accuracy preset — catches every face.
        Uses SCRFD with low thresholds and ensemble fallback."""
        return cls(
            detector="scrfd",
            confidence_threshold=0.3,
            pad_ratio=0.2,
            frame_skip=0,
            temporal_smoothing=True,
            blur_strength=121,
            min_face_size=15,
            track_low_thresh=0.05,
        )
    
    @classmethod
    def privacy_preset(cls) -> "FaceBlurConfig":
        """GDPR-compliant privacy preset — maximum blur, hard to reverse.
        Uses pixelation blur with aggressive face padding."""
        return cls(
            detector="scrfd",
            confidence_threshold=0.3,
            pad_ratio=0.25,
            blur_type="pixelate",
            pixelate_blocks=5,
            frame_skip=0,
            min_face_size=15,
        )
    
    @classmethod
    def realtime_preset(cls) -> "FaceBlurConfig":
        """Real-time processing preset — balanced for live video.
        Uses SCRFD with frame skipping and tracking."""
        return cls(
            detector="scrfd",
            confidence_threshold=0.5,
            frame_skip=2,
            enable_tracking=True,
            temporal_smoothing=True,
            blur_strength=71,
            batch_size=1,
        )
