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
from src.utils.file_utils import ensure_directory, get_project_root
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

        # ── Load Class Weights ──
        # Guards against the indexing mismatch bug that hit us
        # before: raises loudly if class_weights.json's saved
        # order ever drifts from settings.unified_classes again.
        class_weight_dict = self._load_class_weights()

        # ── Phase A: Feature Extraction (frozen base) ──
        logger.info("-" * 60)
        logger.info("PHASE A: Feature Extraction (frozen base)")
        logger.info("-" * 60)

        model = model_builder.build_phase_a()

        phase_a_callbacks = build_callbacks(
            experiment_dir=experiment_dir,
            training_config=self.training_config,
            phase="phase_a"
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.training_config["phase_a_epochs"],
            callbacks=phase_a_callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        plot_training_history(
            csv_path=experiment_dir / "logs" / "phase_a_log.csv",
            model_name=f"{model_name}_phase_a",
            plots_dir=experiment_dir / "plots"
        )

        # ── Reload Best Phase A Weights Before Phase B ──
        # safe_mode=False is required in Keras 3 because the
        # model contains a Lambda layer (resnet50_preprocess /
        # mobilenet preprocess). Without it, load_model() raises
        # a ValueError refusing to deserialize the Lambda.
        phase_a_checkpoint = get_checkpoint_path(
            experiment_dir, "phase_a"
        )
        if phase_a_checkpoint.exists():
            logger.info(
                f"Loading best Phase A weights: "
                f"{phase_a_checkpoint}"
            )
            model = tf.keras.models.load_model(
                str(phase_a_checkpoint),
                safe_mode=False
            )
        else:
            logger.warning(
                "Best Phase A checkpoint not found — "
                "continuing with final epoch weights"
            )

        # ── Phase B: Fine-Tuning ──
        logger.info("-" * 60)
        logger.info("PHASE B: Fine-Tuning (partial unfreeze)")
        logger.info("-" * 60)

        # IMPORTANT: model_builder.base_model refers to the
        # ORIGINAL pre-checkpoint-reload object. After reloading
        # from disk above, `model` contains a NEW nested base
        # model instance. build_phase_b() must unfreeze layers
        # on the base model living INSIDE the reloaded `model`,
        # not the stale `model_builder.base_model` reference —
        # otherwise Phase B silently trains with the base still
        # fully frozen (no error, just wrong results).
        model = model_builder.build_phase_b(model)

        phase_b_callbacks = build_callbacks(
            experiment_dir=experiment_dir,
            training_config=self.training_config,
            phase="phase_b"
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.training_config["phase_b_epochs"],
            callbacks=phase_b_callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        plot_training_history(
            csv_path=experiment_dir / "logs" / "phase_b_log.csv",
            model_name=f"{model_name}_phase_b",
            plots_dir=experiment_dir / "plots"
        )

        # ── Reload Best Phase B Weights Before Evaluation ──
        phase_b_checkpoint = get_checkpoint_path(
            experiment_dir, "phase_b"
        )
        if phase_b_checkpoint.exists():
            logger.info(
                f"Loading best Phase B weights: "
                f"{phase_b_checkpoint}"
            )
            model = tf.keras.models.load_model(
                str(phase_b_checkpoint),
                safe_mode=False
            )
        else:
            logger.warning(
                "Best Phase B checkpoint not found — "
                "continuing with final epoch weights"
            )

        # ── Final Evaluation on Test Set ──
        # This is the ONLY place test data is touched.
        evaluator = ModelEvaluator(
            class_names=self.class_names,
            experiment_dir=experiment_dir
        )
        results = evaluator.evaluate(
            model, test_ds, model_name
        )
        evaluator.save_results(results)

        # ── Save Final Model ──
        final_model_path = (
            experiment_dir / f"{model_name}_final.h5"
        )
        model.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {model_name.upper()}")
        logger.info(f"Experiment dir: {experiment_dir}")
        logger.info("=" * 60)

        return results

    def _save_config_snapshot(self, experiment_dir: Path) -> None:
        """
        Copy the exact config YAML used for this run into the
        experiment directory, so results can always be traced
        back to the settings that produced them.
        """
        config_source = (
            get_project_root() / "configs/preprocessing_config.yaml"
        )
        config_dest = experiment_dir / "config_snapshot.yaml"
        shutil.copy(config_source, config_dest)
        logger.info(f"Config snapshot saved: {config_dest}")

    def _load_class_weights(self) -> Optional[Dict[int, float]]:
        """
        Load precomputed class weights and verify their saved
        class order still matches settings.unified_classes.

        Raises rather than silently training with mismatched
        weights — this is the exact bug class that corrupted
        an earlier run.

        Returns:
            Dict mapping class index -> weight, or None if no
            class_weights.json exists (training proceeds
            unweighted with a warning).
        """
        weights_path = (
            self.settings.splits_output / "class_weights.json"
        )

        if not weights_path.exists():
            logger.warning(
                f"class_weights.json not found at {weights_path} "
                f"— training WITHOUT class weighting"
            )
            return None

        with open(weights_path, "r") as f:
            data = json.load(f)

        saved_order = data["class_order"]
        if saved_order != self.class_names:
            raise ValueError(
                f"class_weights.json class order mismatch!\n"
                f"  Saved order   : {saved_order}\n"
                f"  Expected order: {self.class_names}\n"
                f"Regenerate class_weights.json before training "
                f"— training with mismatched weights would "
                f"silently corrupt results."
            )

        weights_by_idx = {
            int(k): float(v)
            for k, v in data["weights_by_idx"].items()
        }

        logger.info(f"Class weights loaded and verified: {weights_by_idx}")
        return weights_by_idx