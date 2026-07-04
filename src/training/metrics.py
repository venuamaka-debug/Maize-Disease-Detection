# =============================================================
# src/training/metrics.py
# Evaluation metrics for model comparison
# =============================================================
# Calculates all metrics needed for thesis comparison between
# ResNet50 and MobileNetV2. Goes beyond simple accuracy to
# include precision, recall, F1-score per class, confusion
# matrix, and model efficiency metrics.
# =============================================================

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for thesis comparison.

    Calculates and saves:
    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Macro and weighted averages
    - Confusion matrix (saved as PNG)
    - Model size (MB)
    - Inference time per image (ms)
    - Complete results as JSON

    Usage:
        evaluator = ModelEvaluator(class_names, experiment_dir)
        results = evaluator.evaluate(model, test_dataset)
        evaluator.save_results(results)
    """

    def __init__(
        self,
        class_names: List[str],
        experiment_dir: Path
    ):
        """
        Initialize evaluator.

        Args:
            class_names: Ordered list of unified class names
            experiment_dir: Path to save all evaluation outputs
        """
        self.class_names = class_names
        self.experiment_dir = experiment_dir
        self.plots_dir = experiment_dir / "plots"
        ensure_directory(self.plots_dir)
        logger.info("ModelEvaluator initialized")

    def evaluate(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset,
        model_name: str
    ) -> Dict:
        """
        Run complete evaluation on the test dataset.

        This is the ONLY place test data is used.
        Never called during training — only after
        the best model has been selected.

        Args:
            model: Trained Keras model with best weights
            test_dataset: Unseen test tf.data.Dataset
            model_name: "ResNet50" or "MobileNetV2"

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on test set...")
        logger.info(
            "This is the first and only time "
            "test data is used"
        )

        # ── Get Predictions 
        y_true, y_pred, inference_time = (
            self._get_predictions(model, test_dataset)
        )

        # ── Core Metrics ─
        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true, y_pred,
                average=None,
                labels=list(range(len(self.class_names)))
            )
        )

        # Macro averages — treats all classes equally
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # ── Model Size ──
        model_size_mb = self._get_model_size(model)

        # ── Compile Results ─
        results = {
            "model_name": model_name,
            "test_accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "inference_time_ms": float(inference_time),
            "model_size_mb": float(model_size_mb),
            "total_parameters": int(model.count_params()),
            "per_class_metrics": {},
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                output_dict=True
            )
        }

        # Per-class breakdown
        for i, class_name in enumerate(self.class_names):
            results["per_class_metrics"][class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }

        # ── Log Results ─
        self._log_results(results)

        # ── Generate Confusion Matrix ──
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, model_name)

        return results

    def _get_predictions(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run inference on test dataset and measure speed.

        Returns:
            Tuple of (true_labels, predicted_labels,
                     avg_inference_time_ms)
        """
        y_true = []
        y_pred = []

        logger.info("Running inference on test set...")
        start_time = time.time()

        for images, labels in test_dataset:
            predictions = model.predict(
                images, verbose=0
            )
            predicted_classes = np.argmax(
                predictions, axis=1
            )
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes)

        total_time = time.time() - start_time
        n_images = len(y_true)

        # Average inference time per image in milliseconds
        avg_time_ms = (total_time / n_images) * 1000

        logger.info(
            f"Inference complete — "
            f"{n_images} images in "
            f"{total_time:.2f}s "
            f"({avg_time_ms:.2f}ms per image)"
        )

        return (
            np.array(y_true),
            np.array(y_pred),
            avg_time_ms
        )

    def _get_model_size(
        self,
        model: tf.keras.Model
    ) -> float:
        """
        Calculate model file size in megabytes.

        Saves model temporarily to measure actual file size.

        Returns:
            Model size in MB
        """
        temp_path = self.experiment_dir / "temp_model.h5"
        model.save(str(temp_path))
        size_bytes = temp_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        temp_path.unlink()  # Delete temp file
        return size_mb

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str
    ) -> None:
        """
        Generate and save confusion matrix as PNG.

        The confusion matrix shows exactly where the model
        succeeds and fails — which diseases it confuses
        with each other. Essential for thesis analysis.

        Args:
            cm: Confusion matrix array from sklearn
            model_name: Used in title and filename
        """
        # Normalize to percentages
        cm_normalized = (
            cm.astype("float") /
            cm.sum(axis=1)[:, np.newaxis] * 100
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_title(
            f"{model_name} — Confusion Matrix (Counts)"
        )
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")
        axes[0].tick_params(axis="x", rotation=45)

        # Normalized percentages
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1]
        )
        axes[1].set_title(
            f"{model_name} — Confusion Matrix (%)"
        )
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        save_path = (
            self.plots_dir /
            f"confusion_matrix_{model_name.lower()}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved: {save_path}")

    def _log_results(self, results: Dict) -> None:
        """Log evaluation results in a readable format."""
        logger.info("=" * 55)
        logger.info(f"EVALUATION RESULTS: {results['model_name']}")
        logger.info("=" * 55)
        logger.info(
            f"Test Accuracy      : "
            f"{results['test_accuracy']:.4f} "
            f"({results['test_accuracy']*100:.2f}%)"
        )
        logger.info(
            f"Macro Precision    : "
            f"{results['macro_precision']:.4f}"
        )
        logger.info(
            f"Macro Recall       : "
            f"{results['macro_recall']:.4f}"
        )
        logger.info(
            f"Macro F1-Score     : "
            f"{results['macro_f1']:.4f}"
        )
        logger.info(
            f"Model Size         : "
            f"{results['model_size_mb']:.2f} MB"
        )
        logger.info(
            f"Inference Time     : "
            f"{results['inference_time_ms']:.2f} ms/image"
        )
        logger.info(
            f"Total Parameters   : "
            f"{results['total_parameters']:,}"
        )
        logger.info("-" * 55)
        logger.info("Per-Class Results:")
        for cls, m in results["per_class_metrics"].items():
            logger.info(
                f"  {cls:<35} "
                f"P:{m['precision']:.3f} "
                f"R:{m['recall']:.3f} "
                f"F1:{m['f1_score']:.3f}"
            )
        logger.info("=" * 55)

    def save_results(
        self,
        results: Dict,
        filename: str = "results.json"
    ) -> None:
        """
        Save evaluation results to JSON file.

        JSON format means results can be loaded by
        compare_models.py to generate the final
        side-by-side comparison table.

        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        output_path = self.experiment_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {output_path}")


def plot_training_history(
    csv_path: Path,
    model_name: str,
    plots_dir: Path
) -> None:
    """
    Generate accuracy and loss curves from training CSV log.

    Creates two plots side by side:
    - Left: Training vs Validation Accuracy over epochs
    - Right: Training vs Validation Loss over epochs

    These are the standard charts that appear in every
    ML thesis to demonstrate training behaviour.

    Args:
        csv_path: Path to the CSVLogger output file
        model_name: Used in titles and filenames
        plots_dir: Directory to save the PNG
    """
    import pandas as pd

    if not csv_path.exists():
        logger.warning(
            f"Training log not found: {csv_path}"
        )
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy Plot ──
    axes[0].plot(
        df["epoch"], df["accuracy"],
        label="Training", linewidth=2, color="blue"
    )
    axes[0].plot(
        df["epoch"], df["val_accuracy"],
        label="Validation", linewidth=2,
        color="orange", linestyle="--"
    )
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # ── Loss Plot ─
    axes[1].plot(
        df["epoch"], df["loss"],
        label="Training", linewidth=2, color="blue"
    )
    axes[1].plot(
        df["epoch"], df["val_loss"],
        label="Validation", linewidth=2,
        color="orange", linestyle="--"
    )
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"{model_name} Training History",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    save_path = (
        plots_dir /
        f"training_history_{model_name.lower()}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved: {save_path}")