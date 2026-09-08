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
        """
        logger.info("Building ResNet50 — Phase A (frozen base)")

        # ── Input Layer ───────────────────────────────────────
        inputs = tf.keras.Input(
            shape=(*self.image_size, 3),
            name="input_image"
        )

        # ── ResNet50-specific preprocessing ──────────────────
        # Data pipeline normalizes to [0,1]; ResNet50's ImageNet
        # weights expect Caffe-style preprocessing (BGR, mean-
        # subtracted). Without this, the frozen backbone receives
        # out-of-distribution input.
        x = tf.keras.layers.Rescaling(255.0, name="undo_pipeline_norm")(inputs)
        x = tf.keras.layers.Lambda(
            tf.keras.applications.resnet50.preprocess_input,
            name="resnet50_preprocess"
        )(x)

        # ── Load Pre-trained ResNet50 (standalone) ───────────
        self.base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.image_size, 3),
            pooling=None
        )

        # ── Freeze Base Model ─────────────────────────────────
        freeze_base_model(self.base_model)

        # ── Connect Base Model to Inputs ─────────────────────
        base_output = self.base_model(x, training=False)

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

        # ── Locate the base model INSIDE phase_a_model ────────
        # self.base_model is a stale reference after a checkpoint
        # reload creates a new Model object — must find the base
        # model nested inside the model actually passed in here.
        base_model_layer = None
        for layer in phase_a_model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model_layer = layer
                break

        if base_model_layer is None:
            raise RuntimeError(
                "Could not locate the nested ResNet50 base model "
                "inside phase_a_model — check model architecture."
            )

        # ── Unfreeze Top Layers ───────────────────────────────
        unfreeze_top_layers(
            base_model_layer,
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