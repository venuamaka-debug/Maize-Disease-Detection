# =============================================================
# src/scripts/run_preprocessing.py
# Entry point — run the complete preprocessing pipeline
# =============================================================
# This is the ONLY file you execute directly.
# Contains zero business logic — only wires together modules
# and handles command-line arguments.
#
# Usage:
#   python src/scripts/run_preprocessing.py
#   python src/scripts/run_preprocessing.py --mode full
#   python src/scripts/run_preprocessing.py --mode reload
# =============================================================

import sys
import time
import argparse
from pathlib import Path

# Add project root to Python path
# This allows imports like "from src.utils..." to work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_pipeline_logger
from src.config.settings import Settings
from src.data.pipeline import DataPipeline

# Initialize pipeline logger
logger = get_pipeline_logger()


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Maize Leaf Disease Detection System — "
            "Data Preprocessing Pipeline"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (first time)
  python src/scripts/run_preprocessing.py --mode full

  # Reload from saved manifests (subsequent runs)
  python src/scripts/run_preprocessing.py --mode reload

  # Use custom config file
  python src/scripts/run_preprocessing.py --config configs/my_config.yaml
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "reload"],
        default="full",
        help=(
            "Pipeline mode: "
            "'full' runs complete pipeline, "
            "'reload' loads from saved manifests "
            "(default: full)"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing_config.yaml",
        help="Path to YAML config file (default: configs/preprocessing_config.yaml)"
    )

    return parser.parse_args()


def print_banner():
    """Print project banner at startup."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║     MAIZE LEAF DISEASE DETECTION SYSTEM                  ║
║     Data Preprocessing Pipeline v1.0.0                   ║
║     Author: Enuamaka Victor                              ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """
    Main entry point for the preprocessing pipeline.

    Orchestrates the full pipeline execution:
    1. Parse command-line arguments
    2. Load and validate configuration
    3. Run pipeline in specified mode
    4. Report results and timing
    """
    print_banner()
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"Mode   : {args.mode}")
    logger.info(f"Config : {args.config}")

    # Record start time
    start_time = time.time()

    try:
        # ── Load Configuration ───────────────────────────────
        logger.info("Loading configuration...")
        settings = Settings(args.config)
        logger.info(f"✓ Config loaded successfully")

        # ── Initialize Pipeline ──────────────────────────────
        pipeline = DataPipeline(settings)

        # ── Run Pipeline ─────────────────────────────────────
        if args.mode == "full":
            logger.info("Running full pipeline...")
            result = pipeline.run_full()

        elif args.mode == "reload":
            logger.info("Reloading from saved manifests...")
            result = pipeline.run_from_manifests()

        # ── Report Results ───────────────────────────────────
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        logger.info("=" * 60)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"Training samples   : {len(result.manifest.train)}"
        )
        logger.info(
            f"Validation samples : {len(result.manifest.val)}"
        )
        logger.info(
            f"Test samples       : {len(result.manifest.test)}"
        )
        logger.info(
            f"Number of classes  : {result.num_classes}"
        )
        logger.info(
            f"Class names        : {result.class_names}"
        )
        logger.info(
            f"Image size         : {result.image_size}"
        )
        logger.info(
            f"Batch size         : {result.batch_size}"
        )
        logger.info(
            f"Total time         : {minutes}m {seconds}s"
        )
        logger.info("=" * 60)
        logger.info("✓ Pipeline completed successfully!")
        logger.info(
            "Next step: Run model training"
        )
        logger.info("=" * 60)

        return result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(
            "Check that all data paths in config are correct"
        )
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(
            "Check validation errors above and fix data issues"
        )
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()