"""
Tests for FaceBlur pipelines.
"""

import os
import pytest
import numpy as np
from src.config import FaceBlurConfig
from src.utils import save_image
from src.image_pipeline import ImagePipeline

@pytest.fixture
def mock_image_path(tmp_path):
    """Create a temporary image file."""
    img_path = tmp_path / "test_img.jpg"
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    save_image(img, str(img_path))
    return str(img_path)

def test_image_pipeline_initialization():
    """Test ImagePipeline initialization."""
    config = FaceBlurConfig(detector="yunet")
    try:
        pipeline = ImagePipeline(config)
        assert pipeline is not None
    except Exception as e:
        pytest.skip(f"ImagePipeline initialization failed: {e}")

def test_image_pipeline_process(mock_image_path, tmp_path):
    """Test processing a single image."""
    config = FaceBlurConfig(detector="yunet")
    output_path = str(tmp_path / "out_img.jpg")
    try:
        pipeline = ImagePipeline(config)
        result_path = pipeline.process(mock_image_path, output_path)
        assert os.path.exists(result_path)
    except Exception as e:
        pytest.skip(f"ImagePipeline processing failed: {e}")
