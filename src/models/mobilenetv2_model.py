# =============================================================
# src/models/mobilenetv2_model.py
# MobileNetV2 transfer learning model builder
# =============================================================
# Builds the complete MobileNetV2 architecture using ImageNet
# pre-trained weights, with our custom classification head
# attached. Designed for mobile/edge deployment — dramatically
# fewer parameters than ResNet50 while remaining competitive.
# =============================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

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


class MobileNetV2Model:
    """
    MobileNetV2 transfer learning model for maize disease detection.

    MobileNetV2 uses depthwise separable convolutions —
    splitting a standard convolution into two simpler
    operations. This dramatically reduces parameters and
    computation while maintaining competitive accuracy.
    154 layers, ~3.4M parameters — ideal for basic smartphones.

    Usage:
        settings = Settings()
        mobilenet = MobileNetV2Model(settings)

        # Phase A — feature extraction
        model = mobilenet.build_phase_a()

        # Phase B — fine-tuning (after Phase A training)
        model = mobilenet.build_phase_b(model)
    """

    def __init__(self, settings: Settings):
        """
        Initialize the MobileNetV2 model builder.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.image_size = settings.image_size
        self.num_classes = len(settings.unified_classes)
        self.training_config = get_training_config(settings)
        self.base_model = None  # Set when build_phase_a() is called

        logger.info(
            f"MobileNetV2Model initialized — "
            f"input: {self.image_size}, "
            f"classes: {self.num_classes}"
        )

    def build_phase_a(self) -> tf.keras.Model:
        """
        Build MobileNetV2 for Phase A — feature extraction.

        The entire MobileNetV2 base is frozen. Only the new
        classification head is trainable. This stabilizes
        the head before any fine-tuning of pre-trained
        weights happens in Phase B.

        Returns:
            Compiled Keras model ready for Phase A training
        """
        logger.info(
            "Building MobileNetV2 — Phase A (frozen base)"
        )

        # ── Input Layer ───────────────────────────────────────
        inputs = tf.keras.Input(
            shape=(*self.image_size, 3),
            name="input_image"
        )

        # ── Load Pre-trained MobileNetV2 ─────────────────────
        # include_top=False removes ImageNet's 1000-class
        # output layer — we replace it with our own 4-class head
        # weights="imagenet" loads pre-trained knowledge
        self.base_model = MobileNetV2(
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
        # Identical head architecture as ResNet50 — ensures
        # the comparison isolates backbone differences only
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
            name="mobilenetv2_phase_a"
        )

        # ── Compile for Phase A ───────────────────────────────
        model = compile_model(
            model,
            learning_rate=self.training_config["phase_a_lr"],
            optimizer_name=self.training_config["optimizer"]
        )

        log_model_summary(model, "MobileNetV2 — Phase A")
        return model

    def build_phase_b(
        self,
        phase_a_model: tf.keras.Model
    ) -> tf.keras.Model:
        """
        Convert a trained Phase A model into Phase B fine-tuning.

        Unfreezes the top N layers of the MobileNetV2 base
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
        logger.info(
            "Converting MobileNetV2 — Phase A → Phase B"
        )

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

        log_model_summary(phase_a_model, "MobileNetV2 — Phase B")
        return phase_a_model

    def get_model_info(self) -> dict:
        """
        Return metadata about this model architecture.

        Used for logging and the final comparison report.

        Returns:
            Dictionary with model name, parameter counts, etc.
        """
        return {
            "name": "MobileNetV2",
            "total_layers": 154,
            "default_unfreeze": self.training_config[
                "unfreeze_layers"
            ],
            "input_shape": (*self.image_size, 3),
            "num_classes": self.num_classes,
            "pretrained_on": "ImageNet (14M images)",
        }