"""
Tests for FaceBlur pipeline's blur engine.
"""

import pytest
import numpy as np
from src.config import FaceBlurConfig
from src.blur import create_blur_engine
from src.detector import FaceDetection

@pytest.fixture
def mock_image():
    """Create a mock 640x640 image with a white square."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[100:200, 100:200] = 255
    return img

@pytest.fixture
def mock_detection():
    """Create a mock face detection."""
    return FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9)

def test_create_gaussian_blur_engine():
    """Test Gaussian blur engine initialization."""
    config = FaceBlurConfig(blur_type="gaussian", blur_strength=51)
    engine = create_blur_engine(config)
    assert engine.__class__.__name__ == "GaussianBlurEngine"

def test_create_pixelate_blur_engine():
    """Test Pixelation blur engine initialization."""
    config = FaceBlurConfig(blur_type="pixelate", pixelate_blocks=10)
    engine = create_blur_engine(config)
    assert engine.__class__.__name__ == "PixelateBlurEngine"

def test_create_adaptive_blur_engine():
    """Test Adaptive blur engine initialization."""
    config = FaceBlurConfig(blur_type="adaptive", blur_strength=51)
    engine = create_blur_engine(config)
    assert engine.__class__.__name__ == "AdaptiveBlurEngine"

def test_gaussian_blur_application(mock_image, mock_detection):
    """Test applying Gaussian blur."""
    config = FaceBlurConfig(blur_type="gaussian", blur_strength=51, use_elliptical_mask=False)
    engine = create_blur_engine(config)
    
    blurred_img = engine.apply(mock_image, [mock_detection], config)
    
    # Ensure image size is same
    assert blurred_img.shape == mock_image.shape
    
    # Ensure region outside bbox is unchanged
    assert np.array_equal(blurred_img[0:90, 0:90], mock_image[0:90, 0:90])
    
    # Ensure region inside bbox is changed (blurred)
    assert not np.array_equal(blurred_img[100:200, 100:200], mock_image[100:200, 100:200])
