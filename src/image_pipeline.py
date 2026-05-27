"""
Image Processing Pipeline for Face Blur
=========================================

Handles single-image and batch-directory face blurring. This is the simpler
sibling of ``video_pipeline.py`` — no tracking or temporal smoothing needed
because there is only one frame.

Batch mode processes every supported image in a directory tree, logging
per-file results and continuing past individual failures so one corrupted
file doesn't kill an entire run.

Engineering Decisions:
    - Separate ImagePipeline class (vs. reusing VideoPipeline for single-frame
      video) keeps the code path simple and avoids unnecessary VideoCapture
      / VideoWriter overhead for still images.
    - process_batch returns only *successful* output paths, giving the caller
      a reliable list of files that actually exist on disk.
"""

import os
import time
import logging
from typing import List, Optional

import numpy as np

from src.config import FaceBlurConfig
from src.detector import FaceDetection, create_detector
from src.blur import create_blur_engine
from src.visualization import draw_detections, draw_landmarks, draw_info_overlay
from src.utils import (
    validate_input_path,
    get_output_path,
    collect_files,
    load_image,
    save_image,
    Timer,
    IMAGE_EXTENSIONS,
)

# Lazy import tqdm — graceful degradation if not installed
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable or []
            self.n = 0
        def __iter__(self):
            return iter(self._iterable)
        def __next__(self):
            return next(iter(self._iterable))
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass

logger = logging.getLogger(__name__)


class ImagePipeline:
    """Face-blur pipeline for still images.

    Instantiate once with a ``FaceBlurConfig``, then call ``process()`` for
    individual images or ``process_batch()`` for an entire directory.

    Args:
        config: Pipeline configuration controlling detector, blur, and output
                parameters.

    Example:
        >>> config = FaceBlurConfig(detector="scrfd", blur_type="pixelate")
        >>> pipeline = ImagePipeline(config)
        >>> out = pipeline.process("photo.jpg")
        >>> outs = pipeline.process_batch("photos/")
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        self._config = config

        # Initialize components eagerly so errors surface at construction time
        logger.debug("Initializing image pipeline components …")
        self._detector = create_detector(config)
        self._blur_engine = create_blur_engine(config)
        logger.info(
            f"ImagePipeline ready — detector={config.detector}, "
            f"blur={config.blur_type}"
        )

    # ------------------------------------------------------------------
    # Public API — single image
    # ------------------------------------------------------------------

    def process(self, input_path: str, output_path: str = None) -> str:
        """Detect and blur faces in a single image.

        Args:
            input_path: Path to the input image file.
            output_path: Optional explicit output path.  If *None*, an
                         auto-generated path under ``output/`` is used.

        Returns:
            Absolute path to the saved output image.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If *input_path* is not a supported image format.
            IOError: If the image cannot be loaded or saved.
        """
        # ── Step 1: Load image ──────────────────────────────────────
        abs_path, file_type = validate_input_path(input_path)
        if file_type != "image":
            raise ValueError(
                f"Expected an image file, got '{file_type}' for: {abs_path}"
            )

        with Timer("load_image", logger):
            image = load_image(abs_path)

        h, w = image.shape[:2]
        logger.info(f"Loaded image: {abs_path} ({w}x{h})")

        # ── Step 2: Detect faces ────────────────────────────────────
        with Timer("detection", logger) as det_timer:
            detections: List[FaceDetection] = self._detector.detect(image)

        logger.info(
            f"Detected {len(detections)} face(s) in {det_timer.elapsed:.3f}s"
        )

        # ── Step 3: Apply blur ──────────────────────────────────────
        if detections:
            with Timer("blur", logger):
                image = self._blur_engine.apply(image, detections, self._config)

        # ── Step 4: Debug visualizations ────────────────────────────
        if self._config.debug_mode and detections:
            draw_detections(image, detections)
            landmark_dets = [d for d in detections if d.landmarks is not None]
            if landmark_dets:
                draw_landmarks(image, landmark_dets)
            draw_info_overlay(
                image,
                frame_idx=0,
                num_faces=len(detections),
                fps=0.0,
            )

        # ── Step 5: Save output ─────────────────────────────────────
        out = get_output_path(abs_path, output_path)
        saved_path = save_image(image, out, quality=self._config.output_quality)

        # ── Step 6: Log results ─────────────────────────────────────
        logger.info(f"Saved: {saved_path} ({len(detections)} faces blurred)")

        return saved_path

    # ------------------------------------------------------------------
    # Public API — batch directory
    # ------------------------------------------------------------------

    def process_batch(
        self, input_dir: str, output_dir: str = None
    ) -> List[str]:
        """Process all supported images in a directory tree.

        Each image is processed independently. Failures on individual files
        are logged and skipped so one bad file doesn't abort the batch.

        Args:
            input_dir: Root directory to scan for image files.
            output_dir: Optional output directory.  Per-file output paths
                        mirror the input filename with ``_blurred`` suffix.

        Returns:
            List of absolute paths to successfully processed output images.
        """
        # ── Step 1: Collect image files ─────────────────────────────
        abs_dir, dir_type = validate_input_path(input_dir)
        if dir_type != "directory":
            raise ValueError(
                f"Expected a directory, got '{dir_type}' for: {abs_dir}"
            )

        image_files = collect_files(abs_dir, extensions=IMAGE_EXTENSIONS)
        if not image_files:
            logger.warning(f"No supported images found in: {abs_dir}")
            return []

        logger.info(f"Found {len(image_files)} image(s) in {abs_dir}")

        # Ensure output directory exists (if specified)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # ── Step 2-4: Process each image ────────────────────────────
        successful_outputs: List[str] = []
        failed_count: int = 0
        batch_start = time.perf_counter()

        for img_path in tqdm(image_files, desc="Batch processing", unit="img"):
            try:
                # Determine per-file output path
                per_file_output: Optional[str] = None
                if output_dir:
                    per_file_output = os.path.join(
                        output_dir,
                        os.path.splitext(os.path.basename(img_path))[0]
                        + "_blurred"
                        + os.path.splitext(img_path)[1],
                    )

                result = self.process(img_path, per_file_output)
                successful_outputs.append(result)

            except Exception as e:
                # ── Step 4 (cont.): Graceful per-file failure ───────
                failed_count += 1
                logger.error(f"Failed to process {img_path}: {e}")
                continue

        # ── Step 5: Summary ─────────────────────────────────────────
        batch_elapsed = time.perf_counter() - batch_start
        logger.info("=" * 60)
        logger.info("Batch processing complete")
        logger.info(f"  Directory:   {abs_dir}")
        logger.info(f"  Total files: {len(image_files)}")
        logger.info(f"  Succeeded:   {len(successful_outputs)}")
        logger.info(f"  Failed:      {failed_count}")
        logger.info(f"  Total time:  {batch_elapsed:.2f}s")
        logger.info("=" * 60)

        return successful_outputs
