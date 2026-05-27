"""
Utility Module for Face Blur Pipeline
=======================================

Shared helpers for file I/O, format validation, logging setup,
FFmpeg operations, performance timing, and system checks.

This module is dependency-free (uses only stdlib + numpy/cv2)
and provides the infrastructure layer for all other modules.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path
from functools import wraps
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure structured logging for the pipeline.
    
    Args:
        level: Logging verbosity — 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        log_file: Optional path to write logs to file
    
    Returns:
        Configured root logger
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy third-party loggers
    for noisy in ["insightface", "onnxruntime", "mediapipe", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    return logging.getLogger("face_blur")


# ---------------------------------------------------------------------------
# Supported File Formats
# ---------------------------------------------------------------------------

# Formats validated against OpenCV codec support
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_video_file(path: str) -> bool:
    """Check if file has a supported video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    """Check if file has a supported image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_supported_file(path: str) -> bool:
    """Check if file is a supported video or image format."""
    return is_video_file(path) or is_image_file(path)


def validate_input_path(path: str) -> Tuple[str, str]:
    """
    Validate input path exists and is a supported format.
    
    Args:
        path: Path to input file or directory
    
    Returns:
        Tuple of (absolute_path, type) where type is 'video', 'image', or 'directory'
    
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If file format is not supported
    """
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")
    
    if os.path.isdir(path):
        return path, "directory"
    
    if is_video_file(path):
        return path, "video"
    
    if is_image_file(path):
        return path, "image"
    
    raise ValueError(
        f"Unsupported file format: {Path(path).suffix}\n"
        f"Supported video formats: {sorted(VIDEO_EXTENSIONS)}\n"
        f"Supported image formats: {sorted(IMAGE_EXTENSIONS)}"
    )


def get_output_path(
    input_path: str,
    output_path: Optional[str],
    suffix: str = "_blurred"
) -> str:
    """
    Determine output file path.
    
    If output_path is provided, use it directly.
    Otherwise, generate one by appending suffix to input filename.
    
    Args:
        input_path: Original input file path
        output_path: User-specified output path (may be None)
        suffix: Suffix to append if auto-generating path
    
    Returns:
        Absolute path for output file
    """
    if output_path:
        # If output is a directory, place file inside it
        if os.path.isdir(output_path):
            basename = Path(input_path).stem + suffix + Path(input_path).suffix
            return os.path.join(output_path, basename)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        return os.path.abspath(output_path)
    
    # Auto-generate output path in 'output/' directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(input_path).stem + suffix + Path(input_path).suffix
    return os.path.join(output_dir, basename)


def collect_files(directory: str, extensions: set = None) -> List[str]:
    """
    Recursively collect all supported files from a directory.
    
    Args:
        directory: Root directory to scan
        extensions: Set of extensions to include. None = all supported formats.
    
    Returns:
        Sorted list of absolute file paths
    """
    if extensions is None:
        extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                files.append(os.path.join(root, fname))
    
    return sorted(files)


# ---------------------------------------------------------------------------
# FFmpeg Utilities
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible on PATH."""
    return shutil.which("ffmpeg") is not None


def check_ffprobe() -> bool:
    """Check if FFprobe is installed and accessible on PATH."""
    return shutil.which("ffprobe") is not None


def get_video_info(video_path: str) -> dict:
    """
    Extract video metadata using FFprobe.
    
    Falls back to OpenCV if FFprobe is not available.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict with keys: width, height, fps, total_frames, duration,
                        has_audio, codec, bitrate
    """
    info = {}
    
    # Primary: try FFprobe for accurate metadata
    if check_ffprobe():
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                audio_stream = None
                for stream in data.get("streams", []):
                    if stream["codec_type"] == "video" and video_stream is None:
                        video_stream = stream
                    elif stream["codec_type"] == "audio" and audio_stream is None:
                        audio_stream = stream
                
                if video_stream:
                    info["width"] = int(video_stream.get("width", 0))
                    info["height"] = int(video_stream.get("height", 0))
                    
                    # Parse FPS from r_frame_rate (e.g., "30000/1001")
                    fps_str = video_stream.get("r_frame_rate", "30/1")
                    num, den = map(int, fps_str.split("/"))
                    info["fps"] = num / den if den else 30.0
                    
                    info["total_frames"] = int(video_stream.get("nb_frames", 0))
                    info["codec"] = video_stream.get("codec_name", "unknown")
                    info["bitrate"] = int(
                        video_stream.get("bit_rate", 0)
                    ) if video_stream.get("bit_rate") else None
                
                info["has_audio"] = audio_stream is not None
                info["duration"] = float(
                    data.get("format", {}).get("duration", 0)
                )
                
                return info
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"FFprobe failed, falling back to OpenCV: {e}"
            )
    
    # Fallback: OpenCV (less accurate for some metadata)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["duration"] = (
            info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        )
        info["has_audio"] = False  # OpenCV cannot detect audio
        info["codec"] = "unknown"
        info["bitrate"] = None
        cap.release()
    else:
        raise IOError(f"Cannot open video file: {video_path}")
    
    return info


def mux_audio(
    video_no_audio: str,
    original_video: str,
    output_path: str,
    logger: logging.Logger = None
) -> bool:
    """
    Mux (copy) audio from original video into processed video using FFmpeg.
    
    This preserves the original audio track without re-encoding.
    
    Args:
        video_no_audio: Path to processed video (no audio)
        original_video: Path to original video (has audio)
        output_path: Path for final output with audio
        logger: Logger instance
    
    Returns:
        True if audio was successfully muxed, False otherwise
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    if not check_ffmpeg():
        logger.warning(
            "FFmpeg not found. Output video will not have audio. "
            "Install FFmpeg and add to PATH to preserve audio."
        )
        # Just copy the video without audio
        shutil.copy2(video_no_audio, output_path)
        return False
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_no_audio,      # Processed video (no audio)
            "-i", original_video,       # Original video (has audio)
            "-map", "0:v:0",           # Take video from processed
            "-map", "1:a?",            # Take audio from original (? = optional)
            "-c:v", "copy",            # Copy video stream (no re-encode)
            "-c:a", "copy",            # Copy audio stream (no re-encode)
            "-map_metadata", "1",      # Preserve original metadata
            "-shortest",               # Match shortest stream duration
            output_path
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        if result.returncode == 0:
            logger.info("Audio successfully muxed into output video")
            return True
        else:
            logger.warning(f"FFmpeg muxing failed: {result.stderr[:500]}")
            # Fall back to video without audio
            shutil.copy2(video_no_audio, output_path)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg muxing timed out after 300 seconds")
        shutil.copy2(video_no_audio, output_path)
        return False
    except Exception as e:
        logger.error(f"Audio muxing error: {e}")
        shutil.copy2(video_no_audio, output_path)
        return False


# ---------------------------------------------------------------------------
# Performance Timing
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("Face detection") as t:
            results = detector.detect(frame)
        print(f"Detection took {t.elapsed:.3f}s")
    """
    
    def __init__(self, name: str = "", logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0
        self._start = 0.0
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.logger and self.name:
            self.logger.debug(f"{self.name}: {self.elapsed:.4f}s")


def timed(func):
    """
    Decorator to log execution time of functions.
    
    Usage:
        @timed
        def process_frame(frame):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.getLogger(func.__module__).debug(
            f"{func.__qualname__}: {elapsed:.4f}s"
        )
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load an image file into a numpy array (BGR format, as OpenCV convention).
    
    Args:
        path: Path to image file
    
    Returns:
        Image as numpy array in BGR color space
    
    Raises:
        IOError: If image cannot be loaded
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(
            f"Failed to load image: {path}. "
            f"File may be corrupted or in an unsupported format."
        )
    return img


def save_image(
    image: np.ndarray,
    path: str,
    quality: int = 95
) -> str:
    """
    Save numpy array as an image file.
    
    Args:
        image: Image array (BGR)
        path: Output path
        quality: JPEG quality (1-100), PNG compression level mapped from quality
    
    Returns:
        Absolute path of saved file
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    ext = Path(path).suffix.lower()
    params = []
    
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        # Map quality (1-100) to PNG compression (9-0)
        compression = max(0, min(9, int((100 - quality) / 10)))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    
    success = cv2.imwrite(path, image, params)
    if not success:
        raise IOError(f"Failed to save image: {path}")
    
    return os.path.abspath(path)


def resize_for_display(
    image: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720
) -> np.ndarray:
    """Resize image for display while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# System Checks
# ---------------------------------------------------------------------------

def check_system_requirements(logger: logging.Logger = None) -> dict:
    """
    Check system for required and optional dependencies.
    
    Returns:
        Dict with component availability status
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    status = {
        "opencv": False,
        "numpy": False,
        "insightface": False,
        "onnxruntime": False,
        "onnxruntime_gpu": False,
        "mediapipe": False,
        "ffmpeg": False,
        "ffprobe": False,
        "cuda_available": False,
    }
    
    # Core dependencies
    try:
        import cv2
        status["opencv"] = True
        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV not installed!")
    
    try:
        import numpy
        status["numpy"] = True
    except ImportError:
        logger.error("NumPy not installed!")
    
    # Detection dependencies
    try:
        import insightface
        status["insightface"] = True
        logger.info(f"InsightFace: {insightface.__version__}")
    except ImportError:
        logger.warning("InsightFace not installed — SCRFD/RetinaFace unavailable")
    
    try:
        import onnxruntime as ort
        status["onnxruntime"] = True
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            status["onnxruntime_gpu"] = True
            status["cuda_available"] = True
            logger.info(f"ONNX Runtime: {ort.__version__} (GPU available)")
        else:
            logger.info(f"ONNX Runtime: {ort.__version__} (CPU only)")
    except ImportError:
        logger.warning("ONNX Runtime not installed")
    
    try:
        import mediapipe
        status["mediapipe"] = True
        logger.info(f"MediaPipe: {mediapipe.__version__}")
    except ImportError:
        logger.info("MediaPipe not installed (optional)")
    
    # System tools
    status["ffmpeg"] = check_ffmpeg()
    status["ffprobe"] = check_ffprobe()
    if status["ffmpeg"]:
        logger.info("FFmpeg: available")
    else:
        logger.warning("FFmpeg not found — audio preservation disabled")
    
    return status
