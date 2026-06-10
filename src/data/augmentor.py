# =============================================================
# src/data/augmentor.py
# Data augmentation strategy for training images only
# =============================================================
# Defines augmentation transforms applied during training.
# Augmentation is NEVER applied to validation or test data.
# Uses TensorFlow's built-in layers — runs on-the-fly during
# training, not saved to disk, GPU-accelerated when available.
# =============================================================

import tensorflow as tf
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class DataAugmentor:
    """
    Builds and applies augmentation strategy for training.

    Why augmentation matters for this project:
    A farmer photographs a maize leaf in varying conditions —
    different angles, lighting, distances, and orientations.
    Without augmentation, the model only sees clean, centered
    images and fails on real-world variation.

    Augmentation artificially creates this variation during
    training — making the model robust to real conditions.

    Critical rule: ONLY applied to training data.
    Validation and test sets must remain unmodified so
    metrics reflect true real-world performance.

    Usage:
        settings = Settings()
        augmentor = DataAugmentor(settings)
        aug_layer = augmentor.build_augmentation_layer()
    """

    def __init__(self, settings: Settings):
        """
        Initialize augmentor with project settings.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.aug_config = settings.augmentation_config
        self.enabled = settings.augmentation_enabled
        logger.info(
            f"DataAugmentor initialized — "
            f"enabled: {self.enabled}"
        )

    def build_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Build a TensorFlow Sequential augmentation layer.

        Returns a model layer that can be prepended to any
        Keras model. When enabled=False in config, returns
        an identity layer that passes images through unchanged.

        Returns:
            tf.keras.Sequential augmentation pipeline

        Example:
            aug = augmentor.build_augmentation_layer()
            # Use as first layer in your model:
            model = tf.keras.Sequential([aug, base_model, ...])
        """
        if not self.enabled:
            logger.info(
                "Augmentation disabled — "
                "returning identity layer"
            )
            return tf.keras.Sequential(
                [tf.keras.layers.Lambda(lambda x: x)],
                name="augmentation_disabled"
            )

        logger.info("Building augmentation pipeline...")
        layers = []

        # ── Horizontal Flip ──────────────────────────────────
        # Maize leaves can be photographed from either side
        if self.aug_config.get("horizontal_flip", True):
            layers.append(
                tf.keras.layers.RandomFlip("horizontal")
            )
            logger.info("  + RandomFlip (horizontal)")

        # ── Vertical Flip ────────────────────────────────────
        # Disabled by default — upside-down leaves unrealistic
        if self.aug_config.get("vertical_flip", False):
            layers.append(
                tf.keras.layers.RandomFlip("vertical")
            )
            logger.info("  + RandomFlip (vertical)")

        # ── Rotation ─────────────────────────────────────────
        # Farmers photograph leaves at slight angles
        rotation = self.aug_config.get("rotation_range", 15)
        if rotation > 0:
            # Convert degrees to radians for TensorFlow
            rotation_factor = rotation / 360.0
            layers.append(
                tf.keras.layers.RandomRotation(
                    factor=rotation_factor,
                    fill_mode=self.aug_config.get(
                        "fill_mode", "nearest"
                    )
                )
            )
            logger.info(f"  + RandomRotation (±{rotation}°)")

        # ── Zoom ─────────────────────────────────────────────
        # Simulates different camera distances
        zoom = self.aug_config.get("zoom_range", 0.10)
        if zoom > 0:
            layers.append(
                tf.keras.layers.RandomZoom(
                    height_factor=(-zoom, zoom),
                    fill_mode=self.aug_config.get(
                        "fill_mode", "nearest"
                    )
                )
            )
            logger.info(f"  + RandomZoom (±{zoom*100:.0f}%)")

        # ── Width Shift ──────────────────────────────────────
        # Simulates off-center framing
        width_shift = self.aug_config.get(
            "width_shift_range", 0.10
        )
        if width_shift > 0:
            layers.append(
                tf.keras.layers.RandomTranslation(
                    height_factor=0,
                    width_factor=width_shift,
                    fill_mode=self.aug_config.get(
                        "fill_mode", "nearest"
                    )
                )
            )
            logger.info(
                f"  + RandomTranslation width "
                f"(±{width_shift*100:.0f}%)"
            )

        # ── Height Shift ─────────────────────────────────────
        height_shift = self.aug_config.get(
            "height_shift_range", 0.10
        )
        if height_shift > 0:
            layers.append(
                tf.keras.layers.RandomTranslation(
                    height_factor=height_shift,
                    width_factor=0,
                    fill_mode=self.aug_config.get(
                        "fill_mode", "nearest"
                    )
                )
            )
            logger.info(
                f"  + RandomTranslation height "
                f"(±{height_shift*100:.0f}%)"
            )

        # ── Brightness ───────────────────────────────────────
        # Simulates different lighting — morning vs afternoon
        brightness = self.aug_config.get(
            "brightness_range", [0.8, 1.2]
        )
        if brightness:
            # Convert [0.8, 1.2] range to delta format
            delta = (brightness[1] - brightness[0]) / 2
            layers.append(
                tf.keras.layers.RandomBrightness(
                    factor=delta
                )
            )
            logger.info(
                f"  + RandomBrightness "
                f"({brightness[0]}-{brightness[1]})"
            )

        augmentation_pipeline = tf.keras.Sequential(
            layers,
            name="augmentation_pipeline"
        )

        logger.info(
            f"Augmentation pipeline built — "
            f"{len(layers)} transforms"
        )
        return augmentation_pipeline

    def get_augmentation_summary(self) -> dict:
        """
        Return a summary of augmentation settings.

        Useful for logging and documentation.

        Returns:
            Dictionary of active augmentation transforms
        """
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "horizontal_flip": self.aug_config.get(
                "horizontal_flip", True
            ),
            "rotation_range": self.aug_config.get(
                "rotation_range", 15
            ),
            "zoom_range": self.aug_config.get(
                "zoom_range", 0.10
            ),
            "width_shift": self.aug_config.get(
                "width_shift_range", 0.10
            ),
            "height_shift": self.aug_config.get(
                "height_shift_range", 0.10
            ),
            "brightness_range": self.aug_config.get(
                "brightness_range", [0.8, 1.2]
            ),
        }