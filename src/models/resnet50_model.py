# =============================================================
# src/models/resnet50_model.py
# ResNet50 transfer learning model builder
# =============================================================
# Builds the complete ResNet50 architecture using ImageNet
# pre-trained weights, with our custom classification head
# attached. Supports both Phase A (frozen) and Phase B
# (fine-tuned) configurations.
# =============================================================

import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from src.utils.logger import get_logger
from src.config.settings import Settings
from src.models.base_model import (
    build_classification_head,
    compile_model,
    freeze_base_model,
    unfreeze_top_layers,
    log_model_summary,
    get_training_config
)

logger = get_logger(__name__)


class ResNet50Model:
    """
    ResNet50 transfer learning model for maize disease detection.

    ResNet50 uses residual connections (skip connections) that
    allow gradients to flow directly through the network,
    making very deep networks trainable without vanishing
    gradient problems. 50 layers deep, ~25.6M parameters.

    Usage:
        settings = Settings()
        resnet = ResNet50Model(settings)

        # Phase A — feature extraction
        model = resnet.build_phase_a()

        # Phase B — fine-tuning (after Phase A training)
        model = resnet.build_phase_b(model)
    """

    def __init__(self, settings: Settings):
        """
        Initialize the ResNet50 model builder.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.image_size = settings.image_size
        self.num_classes = len(settings.unified_classes)
        self.training_config = get_training_config(settings)
        self.base_model = None  # Set when build_phase_a() is called

        logger.info(
            f"ResNet50Model initialized — "
            f"input: {self.image_size}, "
            f"classes: {self.num_classes}"
        )

    def build_phase_a(self) -> tf.keras.Model:
        """
        Build ResNet50 for Phase A — feature extraction.

        The entire ResNet50 base is frozen. Only the new
        classification head is trainable. This stabilizes
        the head before any fine-tuning of pre-trained
        weights happens in Phase B.

        Returns:
            Compiled Keras model ready for Phase A training
        """
        logger.info("Building ResNet50 — Phase A (frozen base)")

        # ── Input Layer ───────────────────────────────────────
        inputs = tf.keras.Input(
            shape=(*self.image_size, 3),
            name="input_image"
        )

                # ── Load Pre-trained ResNet50 (standalone) ───────────
        # Built WITHOUT input_tensor so we can explicitly call
        # it later with training=False — this is critical for
        # BatchNorm layers to use frozen ImageNet statistics
        # rather than recomputing stats from our small batches.
        self.base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.image_size, 3),
            pooling=None
        )

        # ── Freeze Base Model ─────────────────────────────────
        freeze_base_model(self.base_model)

        # ── Connect Base Model to Inputs ─────────────────────
        # training=False forces BatchNorm layers to always use
        # their pre-trained ImageNet moving statistics, even
        # though the outer model is in training mode during
        # model.fit(). Without this, BatchNorm recalculates
        # statistics from our small maize batches every step,
        # corrupting the pre-trained feature representations.
        base_output = self.base_model(inputs, training=False)

        # ── Attach Classification Head ───────────────────────
        outputs = build_classification_head(
            base_output=base_output,
            num_classes=self.num_classes,
            dense_units=self.training_config["dense_units"],
            dropout_rate=self.training_config["dropout_rate"]
        )

        # ── Build Complete Model ──────────────────────────────
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="resnet50_phase_a"
        )

        # ── Compile for Phase A ───────────────────────────────
        model = compile_model(
            model,
            learning_rate=self.training_config["phase_a_lr"],
            optimizer_name=self.training_config["optimizer"]
        )

        log_model_summary(model, "ResNet50 — Phase A")
        return model

    def build_phase_b(
        self,
        phase_a_model: tf.keras.Model
    ) -> tf.keras.Model:
        """
        Convert a trained Phase A model into Phase B fine-tuning.

        Unfreezes the top N layers of the ResNet50 base
        (configured in YAML) and recompiles with a much
        smaller learning rate. The classification head
        weights from Phase A are preserved — only the
        unfrozen base layers will continue adapting.

        Args:
            phase_a_model: The trained Phase A model
                          (with best weights already loaded)

        Returns:
            Recompiled model ready for Phase B fine-tuning
        """
        logger.info("Converting ResNet50 — Phase A → Phase B")

        # ── Unfreeze Top Layers ───────────────────────────────
        unfreeze_top_layers(
            self.base_model,
            num_layers=self.training_config["unfreeze_layers"]
        )

        # ── Recompile with Smaller Learning Rate ──────────────
        # CRITICAL: Must recompile after changing trainable
        # status, otherwise TensorFlow ignores the change
        phase_a_model = compile_model(
            phase_a_model,
            learning_rate=self.training_config["phase_b_lr"],
            optimizer_name=self.training_config["optimizer"]
        )

        log_model_summary(phase_a_model, "ResNet50 — Phase B")
        return phase_a_model

    def get_model_info(self) -> dict:
        """
        Return metadata about this model architecture.

        Used for logging and the final comparison report.

        Returns:
            Dictionary with model name, parameter counts, etc.
        """
        return {
            "name": "ResNet50",
            "total_layers": 175,
            "default_unfreeze": self.training_config[
                "unfreeze_layers"
            ],
            "input_shape": (*self.image_size, 3),
            "num_classes": self.num_classes,
            "pretrained_on": "ImageNet (14M images)",
        }