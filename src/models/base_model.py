# =============================================================
# src/models/base_model.py
# Shared model utilities used by both ResNet50 and MobileNetV2
# =============================================================
# Contains common functions for building classification heads,
# compiling models, freezing/unfreezing layers, and logging
# model summaries. Guarantees both models are treated
# identically — only the backbone architecture differs.
# =============================================================

import tensorflow as tf
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def build_classification_head(
    base_output: tf.Tensor,
    num_classes: int,
    dense_units: int = 256,
    dropout_rate: float = 0.5
) -> tf.Tensor:
    """
    Build the classification head added on top of base models.

    This identical head is used for both ResNet50 and
    MobileNetV2 — ensuring the comparison between models
    is fair. Only the backbone architecture differs.

    Architecture:
        GlobalAveragePooling2D  → reduces spatial dimensions
        Dense(256, relu)        → learns disease features
        Dropout(0.5)            → prevents overfitting
        Dense(4, softmax)       → outputs class probabilities

    Args:
        base_output: Output tensor from the base model
        num_classes: Number of disease classes (4)
        dense_units: Units in the dense layer (from config)
        dropout_rate: Dropout rate (from config)

    Returns:
        Output tensor with class probabilities
    """
    # GlobalAveragePooling reduces each feature map to
    # a single value — much better than Flatten for
    # transfer learning, less prone to overfitting
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_avg_pool"
    )(base_output)

    # Dense layer learns to combine features
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="dense_features"
    )(x)

    # Dropout randomly deactivates neurons during training
    # Forces the network to learn redundant representations
    # Disabled automatically during inference (predict/evaluate)
    x = tf.keras.layers.Dropout(
        dropout_rate,
        name="dropout"
    )(x)

    # Output layer — one probability per class
    # Softmax ensures all probabilities sum to 1.0
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="output_predictions"
    )(x)

    return outputs


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str = "adam"
) -> tf.keras.Model:
    """
    Compile a model with optimizer, loss, and metrics.

    Used identically for both models and both training phases.
    Only the learning_rate changes between Phase A and Phase B.

    Args:
        model: The Keras model to compile
        learning_rate: Learning rate for the optimizer
        optimizer_name: Optimizer type (default: adam)

    Returns:
        Compiled model ready for training
    """
    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9
        )

    model.compile(
        optimizer=optimizer,
        # Sparse categorical crossentropy expects integer
        # labels (0, 1, 2, 3) not one-hot encoded vectors
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=2, name="top_2_accuracy"
            )
        ]
    )

    logger.info(
        f"Model compiled — "
        f"optimizer: {optimizer_name}, "
        f"lr: {learning_rate}"
    )
    return model


def freeze_base_model(base_model: tf.keras.Model) -> None:
    """
    Freeze all layers in the base model.

    Called at the start of Phase A training.
    Locked weights cannot be modified by backpropagation —
    preserving the ImageNet knowledge while we train
    only the new classification head.

    Args:
        base_model: The pre-trained base model
    """
    base_model.trainable = False
    frozen_count = len(base_model.layers)
    logger.info(
        f"Base model frozen — "
        f"{frozen_count} layers locked"
    )


def unfreeze_top_layers(
    base_model: tf.keras.Model,
    num_layers: int
) -> None:
    """
    Unfreeze the top N layers of the base model for fine-tuning.

    Called at the start of Phase B training.
    Early layers (edges, textures) stay frozen.
    Late layers (complex patterns) are unfrozen to adapt
    to maize disease-specific features.

    Args:
        base_model: The pre-trained base model
        num_layers: Number of layers to unfreeze from the top
    """
    # First freeze everything
    base_model.trainable = True

    # Then freeze all except the last num_layers
    layers_to_freeze = len(base_model.layers) - num_layers

    for layer in base_model.layers[:layers_to_freeze]:
        layer.trainable = False

    # Count trainable vs frozen
    trainable = sum(
        1 for l in base_model.layers if l.trainable
    )
    frozen = len(base_model.layers) - trainable

    logger.info(
        f"Fine-tuning enabled — "
        f"{trainable} layers unfrozen, "
        f"{frozen} layers frozen"
    )


def log_model_summary(
    model: tf.keras.Model,
    model_name: str
) -> None:
    """
    Log a detailed model summary to the logger.

    Reports total parameters, trainable parameters,
    and non-trainable parameters — important for
    comparing model complexity in your thesis.

    Args:
        model: The Keras model to summarize
        model_name: Name for logging identification
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy()
        for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params

    logger.info(f"{'='*50}")
    logger.info(f"Model Summary: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(
        f"Total parameters     : "
        f"{total_params:,}"
    )
    logger.info(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )
    logger.info(
        f"Non-trainable params : "
        f"{non_trainable:,}"
    )
    logger.info(f"{'='*50}")


def get_training_config(settings: Settings) -> dict:
    """
    Extract training configuration from settings.

    Provides a clean interface to all training hyperparameters
    used by both model builders and the trainer.

    Args:
        settings: Loaded Settings instance

    Returns:
        Dictionary with all training hyperparameters
    """
    training = settings.get("training", {})
    return {
        "phase_a_epochs": training.get(
            "phase_a", {}
        ).get("epochs", 20),
        "phase_a_lr": training.get(
            "phase_a", {}
        ).get("learning_rate", 0.001),
        "phase_b_epochs": training.get(
            "phase_b", {}
        ).get("epochs", 30),
        "phase_b_lr": training.get(
            "phase_b", {}
        ).get("learning_rate", 0.0001),
        "unfreeze_layers": training.get(
            "phase_b", {}
        ).get("unfreeze_layers", 30),
        "optimizer": training.get(
            "phase_a", {}
        ).get("optimizer", "adam"),
        "early_stopping_patience": training.get(
            "early_stopping_patience", 10
        ),
        "reduce_lr_patience": training.get(
            "reduce_lr_patience", 5
        ),
        "reduce_lr_factor": training.get(
            "reduce_lr_factor", 0.5
        ),
        "monitor_metric": training.get(
            "monitor_metric", "val_accuracy"
        ),
        "experiments_dir": training.get(
            "experiments_dir", "experiments"
        ),
        "dropout_rate": training.get(
            "dropout_rate", 0.5
        ),
        "dense_units": training.get(
            "dense_units", 256
        ),
    }