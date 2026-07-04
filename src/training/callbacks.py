# =============================================================
# src/training/callbacks.py
# Training callbacks — monitor, save, and control training
# =============================================================
# Defines all Keras callbacks used during model training.
# Callbacks are automatic actions triggered at the end of
# each epoch — saving best weights, stopping early when
# progress stalls, reducing learning rate when needed,
# and logging all metrics to files for analysis.
# =============================================================

import os
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory

logger = get_logger(__name__)


def create_experiment_dir(
    model_name: str,
    experiments_base: str = "experiments"
) -> Path:
    """
    Create a unique timestamped directory for this training run.

    Every run gets its own folder so nothing is ever
    overwritten. You can compare multiple runs and always
    return to any previous result.

    Args:
        model_name: "resnet50" or "mobilenetv2"
        experiments_base: Base experiments directory

    Returns:
        Path to the created experiment directory

    Example:
        experiments/resnet50/run_20260704_153000/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    experiment_dir = (
        Path(experiments_base) / model_name / run_name
    )

    # Create all subdirectories
    ensure_directory(experiment_dir / "checkpoints")
    ensure_directory(experiment_dir / "logs")
    ensure_directory(experiment_dir / "plots")

    logger.info(f"Experiment directory created: {experiment_dir}")
    return experiment_dir


def build_callbacks(
    experiment_dir: Path,
    training_config: dict,
    phase: str = "phase_a"
) -> list:
    """
    Build all training callbacks for one training phase.

    Creates five callbacks that work together to:
    - Save the best model automatically
    - Stop training if progress stalls
    - Reduce learning rate when stuck
    - Log metrics to TensorBoard
    - Save epoch metrics to CSV

    Args:
        experiment_dir: Path to this run's experiment folder
        training_config: Training configuration from settings
        phase: "phase_a" or "phase_b" — affects file naming

    Returns:
        List of configured Keras callbacks
    """
    monitor = training_config.get(
        "monitor_metric", "val_accuracy"
    )
    es_patience = training_config.get(
        "early_stopping_patience", 10
    )
    lr_patience = training_config.get(
        "reduce_lr_patience", 5
    )
    lr_factor = training_config.get(
        "reduce_lr_factor", 0.5
    )

    callbacks = []

    # ── 1. ModelCheckpoint ───────────────────────────────────
    # Saves model weights every time val_accuracy improves.
    # save_best_only=True means only the best epoch is kept.
    # When training ends, this file contains peak performance.
    checkpoint_path = (
        experiment_dir /
        "checkpoints" /
        f"best_{phase}.h5"
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    )
    callbacks.append(checkpoint)
    logger.info(
        f"  ModelCheckpoint → {checkpoint_path.name}"
    )

    # ── 2. EarlyStopping ─────────────────────────────────────
    # Stops training automatically if val_accuracy has not
    # improved for `patience` consecutive epochs.
    # restore_best_weights=True reverts to the best epoch
    # rather than keeping the final (possibly worse) weights.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=es_patience,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    callbacks.append(early_stopping)
    logger.info(
        f"  EarlyStopping → patience={es_patience} epochs"
    )

    # ── 3. ReduceLROnPlateau ─────────────────────────────────
    # Halves the learning rate when val_loss stops improving.
    # Like changing gears — smaller steps when progress slows.
    # This often unlocks additional accuracy improvements
    # after the model appears to have plateaued.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    logger.info(
        f"  ReduceLROnPlateau → "
        f"factor={lr_factor}, patience={lr_patience}"
    )

    # ── 4. TensorBoard ───────────────────────────────────────
    # Writes training metrics to log files that can be
    # visualized in TensorBoard — a browser-based dashboard
    # showing accuracy/loss curves, histograms, and more.
    # Run: tensorboard --logdir experiments/
    tensorboard_dir = experiment_dir / "logs" / phase
    ensure_directory(tensorboard_dir)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch"
    )
    callbacks.append(tensorboard)
    logger.info(
        f"  TensorBoard → {tensorboard_dir}"
    )

    # ── 5. CSVLogger ─────────────────────────────────────────
    # Saves every epoch's metrics to a CSV file.
    # This is where your thesis charts come from —
    # epoch, loss, accuracy, val_loss, val_accuracy per row.
    csv_path = experiment_dir / "logs" / f"{phase}_log.csv"

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_path),
        separator=",",
        append=False
    )
    callbacks.append(csv_logger)
    logger.info(f"  CSVLogger → {csv_path.name}")

    logger.info(
        f"Built {len(callbacks)} callbacks "
        f"for {phase}"
    )
    return callbacks


def get_checkpoint_path(
    experiment_dir: Path,
    phase: str
) -> Path:
    """
    Return the path to the best saved checkpoint.

    Used by the trainer to load best weights before
    starting Phase B or final evaluation.

    Args:
        experiment_dir: Path to this run's experiment folder
        phase: "phase_a" or "phase_b"

    Returns:
        Path to the best checkpoint file
    """
    return experiment_dir / "checkpoints" / f"best_{phase}.h5"