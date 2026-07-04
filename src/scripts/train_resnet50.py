# =============================================================
# src/scripts/train_resnet50.py
# Entry point — train the ResNet50 model
# =============================================================
# Run this script to train ResNet50 from start to finish.
# Reloads data from saved manifests (no re-preprocessing),
# builds the model, runs Phase A and Phase B, evaluates
# on test set, and saves all results.
#
# Usage:
#   python src/scripts/train_resnet50.py
# =============================================================

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.config.settings import Settings
from src.data.pipeline import DataPipeline
from src.models.resnet50_model import ResNet50Model
from src.training.trainer import ModelTrainer

logger = get_logger("train_resnet50")


def main():
    """Train ResNet50 using two-phase transfer learning."""

    print("""
╔══════════════════════════════════════════════════════════╗
║     MAIZE LEAF DISEASE DETECTION SYSTEM                  ║
║     Training: ResNet50                                   ║
║     Author: Enuamaka Victor                              ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Load Configuration ──
    logger.info("Loading configuration...")
    settings = Settings()
    logger.info("✓ Configuration loaded")

    # ── Reload Data From Manifests 
    # Fast reload — skips preprocessing entirely
    # Uses the exact same train/val/test split as before
    logger.info("Loading datasets from saved manifests...")
    pipeline = DataPipeline(settings)
    result = pipeline.run_from_manifests()

    logger.info(
        f"✓ Datasets loaded — "
        f"Train: {len(result.manifest.train)} | "
        f"Val: {len(result.manifest.val)} | "
        f"Test: {len(result.manifest.test)}"
    )

    # ── Build ResNet50 Model ──
    logger.info("Initializing ResNet50 model...")
    resnet = ResNet50Model(settings)

    # Log model info
    info = resnet.get_model_info()
    logger.info(f"  Model          : {info['name']}")
    logger.info(f"  Total layers   : {info['total_layers']}")
    logger.info(f"  Pretrained on  : {info['pretrained_on']}")
    logger.info(
        f"  Unfreeze layers: {info['default_unfreeze']}"
    )

    # ── Train Model ──
    logger.info("Starting two-phase training...")
    trainer = ModelTrainer(settings)

    evaluation_results = trainer.train(
        model_builder=resnet,
        model_name="resnet50",
        train_ds=result.train_dataset,
        val_ds=result.val_dataset,
        test_ds=result.test_dataset
    )

    # ── Final Summary ──
    logger.info("=" * 60)
    logger.info("RESNET50 TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"Test Accuracy  : "
        f"{evaluation_results['test_accuracy']*100:.2f}%"
    )
    logger.info(
        f"Macro F1-Score : "
        f"{evaluation_results['macro_f1']:.4f}"
    )
    logger.info(
        f"Model Size     : "
        f"{evaluation_results['model_size_mb']:.2f} MB"
    )
    logger.info(
        f"Inference Time : "
        f"{evaluation_results['inference_time_ms']:.2f} ms"
    )
    logger.info("=" * 60)
    logger.info(
        "Next: python src/scripts/train_mobilenetv2.py"
    )


if __name__ == "__main__":
    main()