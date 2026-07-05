# =============================================================
# src/scripts/compare_models.py
# Final comparison report — ResNet50 vs MobileNetV2
# =============================================================
# Loads saved results.json from both trained models and
# produces a comprehensive side-by-side comparison.
# This is the academic contribution of your thesis —
# a rigorous, evidence-based comparison of two CNN
# architectures for Nigerian maize disease detection.
#
# Usage:
#   python src/scripts/compare_models.py
# =============================================================

import sys
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory

logger = get_logger("compare_models")


def find_latest_results(
    model_name: str,
    experiments_dir: str = "experiments"
) -> Path:
    """
    Find the most recent results.json for a model.

    Searches the timestamped experiment directories and
    returns the latest run's results file.

    Args:
        model_name: "resnet50" or "mobilenetv2"
        experiments_dir: Base experiments directory

    Returns:
        Path to the latest results.json file

    Raises:
        FileNotFoundError: If no results found for model
    """
    pattern = str(
        Path(experiments_dir) /
        model_name /
        "run_*" /
        "results.json"
    )
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No results found for {model_name}.\n"
            f"Run training first:\n"
            f"  python src/scripts/train_{model_name}.py"
        )

    latest = Path(matches[-1])
    logger.info(
        f"Found {model_name} results: "
        f"{latest.parent.name}"
    )
    return latest


def load_results(results_path: Path) -> dict:
    """
    Load evaluation results from JSON file.

    Args:
        results_path: Path to results.json

    Returns:
        Results dictionary
    """
    with open(results_path, "r") as f:
        return json.load(f)


def print_comparison_table(
    resnet_results: dict,
    mobilenet_results: dict
) -> None:
    """
    Print a formatted side-by-side comparison table.

    This is the core output — the table that goes
    directly into your thesis results chapter.
    """
    r = resnet_results
    m = mobilenet_results

    print("\n")
    print("=" * 65)
    print("  MODEL COMPARISON REPORT")
    print("  Maize Leaf Disease Detection System")
    print("=" * 65)
    print(f"  {'Metric':<30} {'ResNet50':>12} {'MobileNetV2':>12}")
    print("-" * 65)

    # Core metrics
    metrics = [
        ("Test Accuracy (%)",
         r["test_accuracy"] * 100,
         m["test_accuracy"] * 100,
         "{:.2f}%"),
        ("Macro Precision",
         r["macro_precision"],
         m["macro_precision"],
         "{:.4f}"),
        ("Macro Recall",
         r["macro_recall"],
         m["macro_recall"],
         "{:.4f}"),
        ("Macro F1-Score",
         r["macro_f1"],
         m["macro_f1"],
         "{:.4f}"),
        ("Model Size (MB)",
         r["model_size_mb"],
         m["model_size_mb"],
         "{:.2f} MB"),
        ("Inference Time (ms)",
         r["inference_time_ms"],
         m["inference_time_ms"],
         "{:.2f} ms"),
        ("Total Parameters (M)",
         r["total_parameters"] / 1e6,
         m["total_parameters"] / 1e6,
         "{:.2f}M"),
    ]

    for name, r_val, m_val, fmt in metrics:
        r_str = fmt.format(r_val)
        m_str = fmt.format(m_val)
        # Highlight the better value with an arrow
        if name in ["Model Size (MB)", "Inference Time (ms)",
                    "Total Parameters (M)"]:
            # Lower is better
            r_mark = " ←" if r_val < m_val else ""
            m_mark = " ←" if m_val < r_val else ""
        else:
            # Higher is better
            r_mark = " ←" if r_val > m_val else ""
            m_mark = " ←" if m_val > r_val else ""

        print(
            f"  {name:<30} "
            f"{r_str:>12}{r_mark:<3}"
            f"{m_str:>12}{m_mark:<3}"
        )

    print("-" * 65)

    # Per-class F1 scores
    print("  Per-Class F1-Scores:")
    class_names = list(r["per_class_metrics"].keys())
    for cls in class_names:
        r_f1 = r["per_class_metrics"][cls]["f1_score"]
        m_f1 = m["per_class_metrics"][cls]["f1_score"]
        short_name = cls.replace("_", " ")[:28]
        r_mark = " ←" if r_f1 > m_f1 else ""
        m_mark = " ←" if m_f1 > r_f1 else ""
        print(
            f"    {short_name:<28} "
            f"{r_f1:>12.4f}{r_mark:<3}"
            f"{m_f1:>12.4f}{m_mark:<3}"
        )

    print("=" * 65)

    # Recommendation
    print("\n  RECOMMENDATION:")
    print("-" * 65)

    r_acc = r["test_accuracy"]
    m_acc = m["test_accuracy"]
    r_size = r["model_size_mb"]
    m_size = m["model_size_mb"]
    r_speed = r["inference_time_ms"]
    m_speed = m["inference_time_ms"]

    if r_acc > m_acc:
        acc_winner = "ResNet50"
        acc_margin = (r_acc - m_acc) * 100
    else:
        acc_winner = "MobileNetV2"
        acc_margin = (m_acc - r_acc) * 100

    size_reduction = ((r_size - m_size) / r_size) * 100
    speed_improvement = ((r_speed - m_speed) / r_speed) * 100

    print(
        f"  Accuracy:   {acc_winner} is more accurate "
        f"by {acc_margin:.2f}%"
    )
    print(
        f"  Size:       MobileNetV2 is {size_reduction:.1f}% "
        f"smaller than ResNet50"
    )
    print(
        f"  Speed:      MobileNetV2 is {speed_improvement:.1f}% "
        f"faster than ResNet50"
    )
    print()

    if acc_margin < 3.0:
        print(
            "  CONCLUSION: For Nigerian farmers with basic "
            "smartphones,\n"
            "  MobileNetV2 is recommended — competitive "
            "accuracy\n"
            "  with dramatically smaller size and faster "
            "inference."
        )
    else:
        print(
            f"  CONCLUSION: {acc_winner} is recommended "
            f"where accuracy\n"
            f"  is the primary concern. MobileNetV2 remains "
            f"viable\n"
            f"  for resource-constrained deployments."
        )
    print("=" * 65)


def plot_comparison_charts(
    resnet_results: dict,
    mobilenet_results: dict,
    output_dir: Path
) -> None:
    """
    Generate visual comparison charts for thesis.

    Creates three charts:
    1. Core metrics bar chart (accuracy, precision, recall, F1)
    2. Efficiency comparison (size and speed)
    3. Per-class F1 comparison

    Args:
        resnet_results: ResNet50 evaluation results
        mobilenet_results: MobileNetV2 evaluation results
        output_dir: Directory to save charts
    """
    ensure_directory(output_dir)

    colors = {
        "resnet50": "#2196F3",      # Blue
        "mobilenetv2": "#FF9800"    # Orange
    }

    # ── Chart 1: Core Metrics ──
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    r_vals = [
        resnet_results["test_accuracy"],
        resnet_results["macro_precision"],
        resnet_results["macro_recall"],
        resnet_results["macro_f1"]
    ]
    m_vals = [
        mobilenet_results["test_accuracy"],
        mobilenet_results["macro_precision"],
        mobilenet_results["macro_recall"],
        mobilenet_results["macro_f1"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, r_vals, width,
        label="ResNet50",
        color=colors["resnet50"],
        alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, m_vals, width,
        label="MobileNetV2",
        color=colors["mobilenetv2"],
        alpha=0.85
    )

    # Add value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(
        "ResNet50 vs MobileNetV2 — Core Performance Metrics",
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "comparison_core_metrics.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    logger.info("Saved: comparison_core_metrics.png")

    # ── Chart 2: Per-Class F1 ───
    fig, ax = plt.subplots(figsize=(12, 6))

    class_names = list(
        resnet_results["per_class_metrics"].keys()
    )
    short_names = [
        c.replace("_", "\n") for c in class_names
    ]

    r_f1s = [
        resnet_results["per_class_metrics"][c]["f1_score"]
        for c in class_names
    ]
    m_f1s = [
        mobilenet_results["per_class_metrics"][c]["f1_score"]
        for c in class_names
    ]

    x = np.arange(len(class_names))

    bars1 = ax.bar(
        x - width / 2, r_f1s, width,
        label="ResNet50",
        color=colors["resnet50"],
        alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, m_f1s, width,
        label="MobileNetV2",
        color=colors["mobilenetv2"],
        alpha=0.85
    )

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("F1-Score")
    ax.set_title(
        "Per-Class F1-Score Comparison",
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "comparison_per_class_f1.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    logger.info("Saved: comparison_per_class_f1.png")

    # ── Chart 3: Efficiency ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ["ResNet50", "MobileNetV2"]
    sizes = [
        resnet_results["model_size_mb"],
        mobilenet_results["model_size_mb"]
    ]
    speeds = [
        resnet_results["inference_time_ms"],
        mobilenet_results["inference_time_ms"]
    ]
    bar_colors = [
        colors["resnet50"],
        colors["mobilenetv2"]
    ]

    # Model size
    bars = axes[0].bar(
        models, sizes,
        color=bar_colors, alpha=0.85
    )
    for bar, val in zip(bars, sizes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f} MB",
            ha="center", va="bottom", fontweight="bold"
        )
    axes[0].set_title("Model Size (MB)", fontweight="bold")
    axes[0].set_ylabel("Size (MB)")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Inference speed
    bars = axes[1].bar(
        models, speeds,
        color=bar_colors, alpha=0.85
    )
    for bar, val in zip(bars, speeds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f} ms",
            ha="center", va="bottom", fontweight="bold"
        )
    axes[1].set_title(
        "Inference Time per Image (ms)",
        fontweight="bold"
    )
    axes[1].set_ylabel("Time (ms)")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Model Efficiency Comparison",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "comparison_efficiency.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    logger.info("Saved: comparison_efficiency.png")


def main():
    """Run the complete model comparison."""

    print("""
╔══════════════════════════════════════════════════════════╗
║     MAIZE LEAF DISEASE DETECTION SYSTEM                  ║
║     Model Comparison: ResNet50 vs MobileNetV2            ║
║     Author: Enuamaka Victor                              ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Load Results ──
    logger.info("Loading model results...")

    try:
        resnet_path = find_latest_results("resnet50")
        mobilenet_path = find_latest_results("mobilenetv2")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    resnet_results = load_results(resnet_path)
    mobilenet_results = load_results(mobilenet_path)

    logger.info("✓ Both model results loaded")

    # ── Print Comparison Table ──
    print_comparison_table(resnet_results, mobilenet_results)

    # ── Generate Charts ──
    logger.info("Generating comparison charts...")
    output_dir = Path("experiments") / "comparison_charts"
    plot_comparison_charts(
        resnet_results,
        mobilenet_results,
        output_dir
    )

    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"Charts saved to: {output_dir}"
    )
    logger.info(
        "Next: Build the Flask web application"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()