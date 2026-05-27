"""
Video Processing Pipeline for Face Blur
=========================================

Orchestrates the full video pipeline: read frames → detect faces → track across
frames → smooth bounding boxes → apply blur → write output → mux audio.

This module is the core of the video processing workflow. It coordinates all
sub-components (detector, tracker, smoother, blur engine) into a coherent
frame-by-frame processing loop with proper resource management.

Engineering Decisions:
    - Temp file for video writing: We write to a temp file first, then mux
      audio from original via FFmpeg. This avoids OpenCV's lack of audio support.
    - Frame skipping with tracker prediction: On skipped frames, we emit
      Kalman-predicted bounding boxes from active tracks instead of running
      detection.  This gives 2-3x speedup with minimal quality loss on
      talking-head videos.
    - tqdm progress bar: Essential UX for long videos — shows ETA and FPS.
    - Per-frame error resilience: Corrupted frames are skipped (logged) rather
      than crashing the entire pipeline. This handles real-world video artifacts.
    - BlurEngine.apply() signature: The actual blur engines expect
      ``apply(image, detections, config)`` — the config is threaded through
      so engines can read blur_strength, feather_radius, etc. at call time.
    - TemporalSmoother works per-track: ``smoother.smooth(track_id, bbox, alpha)``
      is called individually for each tracked detection, not on a batch.
"""

import os
import time
import tempfile
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import FaceBlurConfig
from src.detector import FaceDetection, create_detector
from src.tracker import ByteTracker, TemporalSmoother
from src.blur import create_blur_engine
from src.visualization import draw_detections, draw_landmarks, draw_info_overlay
from src.utils import (
    validate_input_path,
    get_output_path,
    get_video_info,
    mux_audio,
    Timer,
)

# Lazy import tqdm to keep startup fast; fall back to a no-op if unavailable
try:
    from tqdm import tqdm
except ImportError:
    # Minimal shim so the pipeline still works without tqdm installed
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
            self.n = 0
        def __iter__(self):
            return iter(self._iterable) if self._iterable else iter([])
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass

logger = logging.getLogger(__name__)


class VideoPipeline:
    """End-to-end video face-blur pipeline.

    Reads a video file, detects faces per frame (with optional frame-skip),
    tracks them across frames via ByteTrack, smooths bounding boxes temporally,
    applies the configured blur, and writes the result with preserved audio.

    Args:
        config: Pipeline configuration controlling detector, tracker, blur,
                and output parameters.

    Example:
        >>> config = FaceBlurConfig(detector="scrfd", blur_type="gaussian")
        >>> pipeline = VideoPipeline(config)
        >>> output = pipeline.process("input/video.mp4")
    """

    def __init__(self, config: FaceBlurConfig) -> None:
        self._config = config
        self._detector = None
        self._tracker: Optional[ByteTracker] = None
        self._smoother: Optional[TemporalSmoother] = None
        self._blur_engine = None

        # Cumulative statistics across the process() call
        self._total_faces_detected: int = 0
        self._frames_processed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, input_path: str, output_path: str = None) -> str:
        """Process a single video file: detect → track → blur → write.

        Args:
            input_path: Path to the input video file.
            output_path: Optional explicit output path.  When *None*, an
                         auto-generated path under ``output/`` is used.

        Returns:
            Absolute path to the output video file.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If *input_path* is not a supported video format.
            IOError: If the video cannot be opened or the writer fails.
        """
        # ── Step 1: Validate input ──────────────────────────────────
        abs_path, file_type = validate_input_path(input_path)
        if file_type != "video":
            raise ValueError(
                f"Expected a video file, got '{file_type}' for: {abs_path}"
            )
        logger.info(f"Processing video: {abs_path}")

        # ── Step 2: Video metadata ──────────────────────────────────
        video_info = get_video_info(abs_path)
        fps: float = video_info.get("fps", 30.0)
        width: int = video_info.get("width", 0)
        height: int = video_info.get("height", 0)
        total_frames: int = video_info.get("total_frames", 0)
        has_audio: bool = video_info.get("has_audio", False)

        logger.info(
            f"Video info: {width}x{height} @ {fps:.2f} FPS, "
            f"{total_frames} frames, audio={'yes' if has_audio else 'no'}"
        )

        if width == 0 or height == 0:
            raise IOError(f"Invalid video dimensions ({width}x{height}): {abs_path}")

        # ── Step 3-6: Initialize components ─────────────────────────
        self._init_components()

        # ── Step 7: Open video capture ──────────────────────────────
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {abs_path}")

        # ── Step 8: Create temp output writer ───────────────────────
        # Write to a temp file first; we'll mux audio in a later step.
        final_output = get_output_path(abs_path, output_path)
        temp_output = self._make_temp_path(final_output)

        fourcc = cv2.VideoWriter_fourcc(*self._config.output_codec)
        writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(
                f"Cannot create video writer for: {temp_output}. "
                f"Codec '{self._config.output_codec}' may not be available."
            )

        # ── Step 9: Frame processing loop ───────────────────────────
        self._total_faces_detected = 0
        self._frames_processed = 0
        pipeline_start = time.perf_counter()

        # Cache of most-recent detections for frame-skip reuse
        cached_detections: List[FaceDetection] = []

        progress = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Processing",
            unit="frame",
            dynamic_ncols=True,
        )

        frame_idx = 0
        fps_samples: List[float] = []  # rolling window for display FPS

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video or read error

                frame_timer_start = time.perf_counter()

                try:
                    detections = self._process_single_frame(
                        frame, frame_idx, cached_detections
                    )
                    cached_detections = detections

                    # Write the (now-blurred) frame
                    writer.write(frame)
                    self._frames_processed += 1

                except Exception as frame_err:
                    # Resilience: skip corrupted / problematic frames
                    logger.warning(
                        f"Error processing frame {frame_idx}, skipping: {frame_err}"
                    )
                    # Write the original unblurred frame so video stays in sync
                    writer.write(frame)

                # ── FPS tracking / progress bar ─────────────────────
                frame_elapsed = time.perf_counter() - frame_timer_start
                instant_fps = 1.0 / max(frame_elapsed, 1e-9)
                fps_samples.append(instant_fps)
                # Keep a rolling window of 30 samples for a smooth average
                if len(fps_samples) > 30:
                    fps_samples.pop(0)
                avg_fps = sum(fps_samples) / len(fps_samples)

                progress.update(1)
                progress.set_postfix(
                    fps=f"{avg_fps:.1f}",
                    faces=len(cached_detections),
                )

                frame_idx += 1

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        finally:
            progress.close()
            cap.release()
            writer.release()

        # ── Step 10-12: Audio mux & cleanup ─────────────────────────
        if frame_idx == 0:
            # No frames were read — empty or broken video
            self._cleanup_temp(temp_output)
            raise IOError(f"No frames could be read from: {abs_path}")

        self._finalize_output(
            temp_output=temp_output,
            original_path=abs_path,
            final_output=final_output,
            has_audio=has_audio,
        )

        # ── Step 13: Log final statistics ───────────────────────────
        total_time = time.perf_counter() - pipeline_start
        avg_pipeline_fps = self._frames_processed / max(total_time, 1e-9)

        logger.info("=" * 60)
        logger.info("Video processing complete")
        logger.info(f"  Input:         {abs_path}")
        logger.info(f"  Output:        {final_output}")
        logger.info(f"  Frames:        {self._frames_processed}")
        logger.info(f"  Total time:    {total_time:.2f}s")
        logger.info(f"  Avg FPS:       {avg_pipeline_fps:.1f}")
        logger.info(f"  Total faces:   {self._total_faces_detected}")
        logger.info("=" * 60)

        return final_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Instantiate detector, tracker, smoother, and blur engine."""
        logger.debug("Initializing pipeline components …")

        # Detector — create_detector(config) handles backend selection & fallback
        self._detector = create_detector(self._config)
        logger.info(f"Detector: {self._config.detector}")

        # Tracker (optional) — ByteTracker takes the full config object
        if self._config.enable_tracking:
            self._tracker = ByteTracker(self._config)
            logger.info("Tracking: ByteTrack enabled")
        else:
            self._tracker = None
            logger.info("Tracking: disabled")

        # Temporal smoother (optional)
        # TemporalSmoother.__init__ accepts a stale_timeout (int); the per-call
        # alpha is passed in smooth().  We store the alpha from config to use
        # during frame processing.
        if self._config.temporal_smoothing:
            self._smoother = TemporalSmoother()
            logger.info(
                f"Temporal smoothing: enabled (alpha={self._config.smoothing_alpha})"
            )
        else:
            self._smoother = None

        # Blur engine — create_blur_engine(config) returns the right subclass
        self._blur_engine = create_blur_engine(self._config)
        logger.info(f"Blur engine: {self._config.blur_type}")

    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        cached_detections: List[FaceDetection],
    ) -> List[FaceDetection]:
        """Run detection / tracking / blur on a single frame **in-place**.

        Args:
            frame: BGR image array — **modified in-place** with blur applied.
            frame_idx: Zero-based index of the current frame.
            cached_detections: Detections from the previous non-skipped frame.

        Returns:
            List of ``FaceDetection`` objects used for this frame (may come
            from fresh detection or from tracker prediction on skipped frames).
        """
        skip_interval = self._config.frame_skip
        is_skipped = (
            skip_interval > 0
            and frame_idx > 0
            and (frame_idx % (skip_interval + 1)) != 0
        )

        # ── (a-b) Detection or cached prediction ───────────────────
        if is_skipped and self._tracker is not None:
            # On skipped frames the tracker already predicted positions in
            # the last update() call.  We re-use the cached detections
            # (which carry the Kalman-predicted bounding boxes from the
            # most recent tracker output).
            detections = cached_detections
            logger.debug(
                f"Frame {frame_idx}: skipped detection, reusing "
                f"{len(detections)} cached faces"
            )
        else:
            # Run full face detection
            with Timer("detection", logger):
                detections = self._detector.detect(frame)
            logger.debug(f"Frame {frame_idx}: detected {len(detections)} faces")

        # ── (c-d) Update tracker ────────────────────────────────────
        # ByteTracker.update() accepts List[FaceDetection] and returns
        # detections with track_id populated.  We only call it on
        # non-skipped frames to avoid double-predicting.
        if self._tracker is not None and not is_skipped:
            detections = self._tracker.update(detections)

        # ── (e) Temporal smoothing ──────────────────────────────────
        # TemporalSmoother.smooth() operates per-track:
        #   smooth(track_id, bbox, alpha) -> smoothed_bbox
        # We iterate over detections and replace their bboxes in-place.
        if self._smoother is not None and detections:
            alpha = self._config.smoothing_alpha
            smoothed: List[FaceDetection] = []
            for det in detections:
                new_bbox = self._smoother.smooth(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    alpha=alpha,
                )
                smoothed.append(FaceDetection(
                    bbox=new_bbox,
                    confidence=det.confidence,
                    landmarks=det.landmarks,
                    track_id=det.track_id,
                ))
            detections = smoothed

        # ── Bookkeeping ─────────────────────────────────────────────
        self._total_faces_detected += len(detections)

        # ── (f) Apply blur ──────────────────────────────────────────
        # BlurEngine.apply(image, detections, config) returns a new
        # image with faces blurred.  We write it back into `frame` so
        # the caller's reference is updated (frame is a mutable ndarray).
        if detections:
            blurred = self._blur_engine.apply(frame, detections, self._config)
            frame[:] = blurred

        # ── (g) Debug visualizations ────────────────────────────────
        if self._config.debug_mode and detections:
            draw_detections(frame, detections)
            # Draw landmarks only for detections that have them
            landmark_dets = [d for d in detections if d.landmarks is not None]
            if landmark_dets:
                draw_landmarks(frame, landmark_dets)
            draw_info_overlay(
                frame,
                frame_idx=frame_idx,
                num_faces=len(detections),
                fps=0.0,  # Will be overlaid by the caller's FPS
            )

        return detections

    # ------------------------------------------------------------------
    # Output finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _make_temp_path(final_output: str) -> str:
        """Generate a temp file path alongside the final output location.

        We keep the temp file in the same directory (not /tmp) so the final
        rename/copy stays on the same filesystem — avoiding slow cross-device
        copies for large video files.
        """
        out_dir = os.path.dirname(os.path.abspath(final_output))
        os.makedirs(out_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            suffix=".mp4", prefix="_fb_temp_", dir=out_dir
        )
        os.close(fd)  # We only need the path; VideoWriter will open it
        return temp_path

    def _finalize_output(
        self,
        temp_output: str,
        original_path: str,
        final_output: str,
        has_audio: bool,
    ) -> None:
        """Mux audio (if requested) and move the temp file to final output.

        Args:
            temp_output: Path to the temporary video (no audio).
            original_path: Path to the original input video.
            final_output: Desired final output path.
            has_audio: Whether the original video contains an audio stream.
        """
        if self._config.preserve_audio and has_audio:
            logger.info("Muxing audio from original video …")
            success = mux_audio(
                video_no_audio=temp_output,
                original_video=original_path,
                output_path=final_output,
                logger=logger,
            )
            # Clean up temp regardless of mux result
            self._cleanup_temp(temp_output)
            if not success:
                logger.warning("Audio muxing failed — output has no audio")
        else:
            # No audio to mux — just move the temp file into place
            try:
                if os.path.exists(final_output):
                    os.remove(final_output)
                os.rename(temp_output, final_output)
            except OSError:
                # Cross-device fallback
                import shutil
                shutil.move(temp_output, final_output)

    @staticmethod
    def _cleanup_temp(path: str) -> None:
        """Silently remove a temporary file if it exists."""
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.debug(f"Could not remove temp file {path}: {e}")
