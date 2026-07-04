# =============================================================
# src/training/trainer.py
# Training orchestrator — executes two-phase training
# =============================================================
# Coordinates the complete training process for any model.
# Handles Phase A (frozen base) and Phase B (fine-tuning),
# saves config snapshots, generates plots, and produces
# the final evaluation report. Used identically for both
# ResNet50 and MobileNetV2 — only the model differs.
# =============================================================

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory
from src.utils.reproducibility import set_global_seeds
from src.config.settings import Settings
from src.models.base_model import get_training_config
from src.training.callbacks import (
    create_experiment_dir,
    build_callbacks,
    get_checkpoint_path
)
from src.training.metrics import (
    ModelEvaluator,
    plot_training_history
)

logger = get_logger(__name__)


class ModelTrainer:
    """
    Executes the complete two-phase training strategy.

    Handles everything after the model is built:
    - Phase A: Train classification head (frozen base)
    - Phase B: Fine-tune top layers (partial unfreeze)
    - Saving best checkpoints at each phase
    - Generating training curves and evaluation plots
    - Saving complete results to JSON

    The same Trainer is used for both ResNet50 and
    MobileNetV2 — the model is passed in as a parameter.

    Usage:
        settings = Settings()
        trainer = ModelTrainer(settings)

        resnet = ResNet50Model(settings)
        model_a = resnet.build_phase_a()

        result = trainer.train(
            model_builder=resnet,
            model_name="resnet50",
            train_ds=train_dataset,
            val_ds=val_dataset,
            test_ds=test_dataset
        )
    """

    def __init__(self, settings: Settings):
        """
        Initialize trainer with project settings.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.training_config = get_training_config(settings)
        self.class_names = settings.unified_classes
        logger.info("ModelTrainer initialized")
        logger.info(
            f"  Phase A: "
            f"{self.training_config['phase_a_epochs']} epochs, "
            f"lr={self.training_config['phase_a_lr']}"
        )
        logger.info(
            f"  Phase B: "
            f"{self.training_config['phase_b_epochs']} epochs, "
            f"lr={self.training_config['phase_b_lr']}"
        )

    def train(
        self,
        model_builder,
        model_name: str,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset
    ) -> Dict:
        """
        Execute the complete two-phase training pipeline.

        Args:
            model_builder: ResNet50Model or MobileNetV2Model
            model_name: "resnet50" or "mobilenetv2"
            train_ds: Training tf.data.Dataset
            val_ds: Validation tf.data.Dataset
            test_ds: Test tf.data.Dataset (used ONLY at end)

        Returns:
            Complete evaluation results dictionary
        """
        logger.info("=" * 60)
        logger.info(f"STARTING TRAINING: {model_name.upper()}")
        logger.info("=" * 60)

        # ── Set Seeds for Reproducibility ──
        set_global_seeds(self.settings.seed)

        # ── Create Experiment Directory ─
        experiment_dir = create_experiment_dir(
            model_name=model_name,
            experiments_base=self.training_config[
                "experiments_dir"
            ]
        )

        # ── Save Config Snapshot ─
        # Records exactly what settings produced these results
        self._save_config_snapshot(experiment_dir)

        # ── Phase A: Feature Extraction