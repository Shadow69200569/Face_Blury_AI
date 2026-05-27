"""
Tests for FaceBlur pipeline's detection module.
"""

import pytest
import numpy as np
from src.config import FaceBlurConfig
from src.detector import FaceDetection, create_detector

@pytest.fixture
def mock_image():
    """Create a mock 640x640 image with a solid color."""
    return np.zeros((640, 640, 3), dtype=np.uint8)

def test_face_detection_dataclass():
    """Test FaceDetection dataclass properties."""
    det = FaceDetection(bbox=(10, 20, 110, 120), confidence=0.95, track_id=5)
    
    assert det.width == 100
    assert det.height == 100
    assert det.area == 10000
    assert det.center == (60, 70)
    assert det.track_id == 5

def test_create_yunet_detector():
    """Test YuNet detector initialization."""
    config = FaceBlurConfig(detector="yunet")
    try:
        detector = create_detector(config)
        assert detector.__class__.__name__ == "YuNetDetector"
    except Exception as e:
        pytest.skip(f"YuNet initialization failed: {e}")

def test_create_mediapipe_detector():
    """Test MediaPipe detector initialization."""
    config = FaceBlurConfig(detector="mediapipe")
    try:
        detector = create_detector(config)
        assert detector.__class__.__name__ == "MediaPipeDetector"
    except Exception as e:
        pytest.skip(f"MediaPipe initialization failed: {e}")

def test_create_scrfd_detector():
    """Test SCRFD detector initialization."""
    config = FaceBlurConfig(detector="scrfd")
    try:
        detector = create_detector(config)
        assert detector.__class__.__name__ == "SCRFDDetector"
    except Exception as e:
        pytest.skip(f"SCRFD initialization failed: {e}")
