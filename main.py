#!/usr/bin/env python3
"""
Face Blur Pipeline — CLI Entry Point
======================================

Command-line interface for the face blur pipeline.  Supports processing
single video files, single images, and entire directories of mixed media.

Usage examples:
    # Single video (auto-detect output path)
    python main.py -i input/video.mp4

    # Single image with pixelate blur
    python main.py -i photo.jpg --blur pixelate

    # Directory batch processing
    python main.py -i input/ -o output/

    # Privacy-focused preset
    python main.py -i interview.mp4 --preset privacy

    # Debug mode with realtime preset
    python main.py -i crowd.mp4 --preset realtime --debug

    # System check only
    python main.py --check-system

Engineering Decisions:
    - argparse over click/typer: Zero extra dependencies for a CLI that
      only needs flat flags — no sub-commands or complex nesting.
    - Presets override individual args: A preset sets a coherent group of
      parameters; individual flags can still override specific fields after.
    - Auto-detection of input type (video / image / directory) removes
      cognitive load from the user — they just point at a path.
    - KeyboardInterrupt is caught at the top level for a clean exit
      (no traceback) when the user hits Ctrl+C during long runs.
"""

import sys
import os
import time
import argparse
import logging
from typing import List

from src.config import FaceBlurConfig
from src.utils import (
    setup_logging,
    validate_input_path,
    check_system_requirements,
    is_video_file,
    is_image_file,
    collect_files,
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
)
from src.video_pipeline import VideoPipeline
from src.image_pipeline import ImagePipeline

logger: logging.Logger  # Initialized in main()


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="face_blur",
        description=(
            "Production-quality face detection and blurring for videos and "
            "images.  Supports multiple detection backends, blur styles, "
            "ByteTrack tracking, and audio preservation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py -i video.mp4\n"
            "  python main.py -i photo.jpg --blur pixelate\n"
            "  python main.py -i input_dir/ -o output_dir/\n"
            "  python main.py -i interview.mp4 --preset privacy\n"
            "  python main.py --check-system\n"
        ),
    )

    # ── Required ────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input file (video / image) or directory.  Required unless --check-system is used.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file or directory.  Auto-generated if omitted.",
    )

    # ── Detection ───────────────────────────────────────────────────
    det_group = parser.add_argument_group("Detection")
    det_group.add_argument(
        "--detector",
        type=str,
        choices=["scrfd", "retinaface", "yunet", "mediapipe", "ensemble"],
        default="scrfd",
        help="Face detection backend (default: scrfd).",
    )
    det_group.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence 0.0–1.0 (default: 0.5).",
    )

    # ── Blur ────────────────────────────────────────────────────────
    blur_group = parser.add_argument_group("Blur")
    blur_group.add_argument(
        "--blur",
        type=str,
        choices=["gaussian", "pixelate", "adaptive"],
        default="gaussian",
        help="Blur method (default: gaussian).",
    )
    blur_group.add_argument(
        "--blur-strength",
        type=int,
        default=99,
        help="Gaussian blur kernel size — must be odd (default: 99).",
    )

    # ── Video ───────────────────────────────────────────────────────
    vid_group = parser.add_argument_group("Video")
    vid_group.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Detect every N+1 frames; use tracker on skipped frames (default: 0 = every frame).",
    )
    vid_group.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable ByteTrack face tracking.",
    )
    vid_group.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not preserve audio in the output video.",
    )

    # ── Presets ─────────────────────────────────────────────────────
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "accurate", "privacy", "realtime"],
        default=None,
        help="Use a preset configuration.  Individual flags override preset values.",
    )

    # ── Debug & system ──────────────────────────────────────────────
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug visualization (bounding boxes, landmarks, track IDs).",
    )
    sys_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device (default: auto-detect).",
    )
    sys_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Print detailed performance statistics after processing.",
    )
    sys_group.add_argument(
        "--check-system",
        action="store_true",
        help="Verify system requirements and exit.",
    )
    sys_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return parser


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> FaceBlurConfig:
    """Build a ``FaceBlurConfig`` from parsed CLI arguments.

    If a ``--preset`` is specified, it provides the base configuration.
    Individual CLI flags override the preset values.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Fully populated ``FaceBlurConfig``.
    """
    # Start from preset (if given), otherwise use plain defaults
    preset_map = {
        "fast": FaceBlurConfig.fast_preset,
        "accurate": FaceBlurConfig.accurate_preset,
        "privacy": FaceBlurConfig.privacy_preset,
        "realtime": FaceBlurConfig.realtime_preset,
    }

    if args.preset:
        config = preset_map[args.preset]()
        logger.info(f"Using preset: {args.preset}")
    else:
        config = FaceBlurConfig()

    # Override with explicit CLI flags.  We only override if the user actually
    # passed the flag (i.e. the value differs from the parser default *or*
    # the user explicitly typed it).  For simplicity we always apply — presets
    # are the base and CLI args are the overlay.
    config.detector = args.detector
    config.confidence_threshold = args.confidence
    config.blur_type = args.blur
    config.blur_strength = args.blur_strength
    config.frame_skip = args.frame_skip
    config.device = args.device
    config.debug_mode = args.debug
    config.log_level = args.log_level

    # Boolean flags (store_true → only override when True)
    if args.no_tracking:
        config.enable_tracking = False
    if args.no_audio:
        config.preserve_audio = False

    return config


# ---------------------------------------------------------------------------
# Main processing orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace, config: FaceBlurConfig) -> None:
    """Route input to the appropriate pipeline based on file type.

    Handles single files (video or image) and directories (batch mode).

    Args:
        args: Parsed CLI arguments (for input/output paths).
        config: Validated pipeline configuration.
    """
    input_path: str = args.input
    output_path: str = args.output

    abs_path, input_type = validate_input_path(input_path)
    start_time = time.perf_counter()

    if input_type == "video":
        # ── Single video file ───────────────────────────────────────
        pipeline = VideoPipeline(config)
        result = pipeline.process(abs_path, output_path)
        _print_summary([result], start_time, args.benchmark)

    elif input_type == "image":
        # ── Single image file ───────────────────────────────────────
        pipeline = ImagePipeline(config)
        result = pipeline.process(abs_path, output_path)
        _print_summary([result], start_time, args.benchmark)

    elif input_type == "directory":
        # ── Directory batch mode ────────────────────────────────────
        all_files = collect_files(abs_path)
        if not all_files:
            logger.warning(f"No supported files found in: {abs_path}")
            return

        # Separate videos and images for their respective pipelines
        video_files = [f for f in all_files if is_video_file(f)]
        image_files = [f for f in all_files if is_image_file(f)]

        logger.info(
            f"Directory scan: {len(video_files)} video(s), "
            f"{len(image_files)} image(s)"
        )

        results: List[str] = []

        # Process videos
        if video_files:
            vid_pipeline = VideoPipeline(config)
            for vf in video_files:
                try:
                    result = vid_pipeline.process(vf, output_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process video {vf}: {e}")

        # Process images (batch)
        if image_files:
            img_pipeline = ImagePipeline(config)
            img_results = img_pipeline.process_batch(abs_path, output_path)
            results.extend(img_results)

        _print_summary(results, start_time, args.benchmark)
    else:
        logger.error(f"Unknown input type '{input_type}' for: {abs_path}")


def _print_summary(
    outputs: List[str],
    start_time: float,
    benchmark: bool,
) -> None:
    """Print a human-readable summary to the console.

    Args:
        outputs: List of output file paths produced.
        start_time: ``time.perf_counter()`` timestamp from before processing.
        benchmark: If True, print extended performance metrics.
    """
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("  Face Blur Pipeline — Complete")
    print("=" * 60)
    print(f"  Files processed: {len(outputs)}")
    print(f"  Total time:      {elapsed:.2f}s")

    if outputs:
        print(f"  Output(s):")
        for o in outputs:
            size_mb = os.path.getsize(o) / (1024 * 1024) if os.path.isfile(o) else 0
            print(f"    • {o}  ({size_mb:.1f} MB)")

    if benchmark:
        print(f"\n  Performance:")
        print(f"    Wall time:     {elapsed:.3f}s")
        print(f"    Avg per file:  {elapsed / max(len(outputs), 1):.3f}s")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point.  Parses arguments and runs the pipeline."""
    global logger

    parser = build_parser()
    args = parser.parse_args()

    # ── System check mode (no input required) ───────────────────────
    if args.check_system:
        root_logger = setup_logging(level=args.log_level)
        logger = logging.getLogger(__name__)
        print("\n🔍 Checking system requirements …\n")
        status = check_system_requirements(logger=root_logger)
        print("\nComponent Status:")
        for component, available in status.items():
            icon = "✅" if available else "❌"
            print(f"  {icon}  {component}")
        print()
        return

    # ── Validate that --input was provided ──────────────────────────
    if args.input is None:
        parser.error("--input / -i is required (unless using --check-system)")

    # ── Logging setup ───────────────────────────────────────────────
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # ── Build configuration ─────────────────────────────────────────
    try:
        config = build_config(args)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    if args.debug:
        logger.info("Debug mode enabled — output will contain visual overlays")

    # ── Run ─────────────────────────────────────────────────────────
    try:
        run_pipeline(args, config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user (Ctrl+C).  Exiting cleanly.\n")
        sys.exit(130)  # Standard SIGINT exit code
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
