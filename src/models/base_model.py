# =============================================================
# src/models/base_model.py
# Shared model utilities used by both ResNet50 and MobileNetV2
# =============================================================

import tensorflow as tf
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def build_classification_head(
    base_output: tf.Tensor,
    num_classes: int,
    dense_units: int = 256,
    dropout_rate: float = 0.6
) -> tf.Tensor:
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_avg_pool"
    )(base_output)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(
        dropout_rate,
        name="dropout"
    )(x)

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
    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9
        )

    model.compile(
        optimizer=optimizer,
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
    base_model.trainable = True
    layers_to_freeze = len(base_model.layers) - num_layers

    for layer in base_model.layers[:layers_to_freeze]:
        layer.trainable = False

    bn_frozen_count = 0
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn_frozen_count += 1

    trainable = sum(
        1 for l in base_model.layers if l.trainable
    )
    frozen = len(base_model.layers) - trainable

    logger.info(
        f"Fine-tuning enabled — "
        f"{trainable} layers unfrozen, "
        f"{frozen} layers frozen "
        f"({bn_frozen_count} BatchNorm layers kept frozen)"
    )


def log_model_summary(
    model: tf.keras.Model,
    model_name: str
) -> None:
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
            "early_stopping_patience", 15
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
