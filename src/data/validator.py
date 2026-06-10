# =============================================================
# src/data/validator.py
# Raw dataset validation before any processing begins
# =============================================================
# Inspects both datasets and reports problems before we waste
# time preprocessing bad data. Validates folder structure,
# image counts, file integrity, and class balance.
# Never modifies data — pure inspection only.
# =============================================================

import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.file_utils import (
    list_image_files,
    count_images_per_class,
    is_valid_image
)
from src.config.settings import Settings

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """
    Structured report produced after dataset validation.

    Contains counts, warnings, errors, and a final verdict
    on whether the dataset is safe to process.
    """
    total_images: int = 0
    images_per_class: Dict[str, int] = field(default_factory=dict)
    corrupted_files: List[str] = field(default_factory=list)
    missing_folders: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = False

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            f"Total images found : {self.total_images}",
            f"Corrupted files    : {len(self.corrupted_files)}",
            f"Missing folders    : {len(self.missing_folders)}",
            f"Warnings           : {len(self.warnings)}",
            f"Errors             : {len(self.errors)}",
            "-" * 60,
            "Images per class:",
        ]
        for cls, count in self.images_per_class.items():
            lines.append(f"  {cls:<35} {count:>6} images")
        lines.append("-" * 60)
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        lines.append("-" * 60)
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines.append(f"Validation Status  : {status}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DatasetValidator:
    """
    Validates raw datasets before preprocessing begins.

    Checks both PlantVillage and Mendeley datasets for:
    - Correct folder structure
    - Minimum image counts per class
    - Corrupted or unreadable image files
    - Class imbalance issues
    - Overall dataset integrity

    Usage:
        settings = Settings()
        validator = DatasetValidator(settings)
        report = validator.validate_all()
        if not report.passed:
            raise RuntimeError("Dataset validation failed")
    """

    def __init__(self, settings: Settings):
        """
        Initialize validator with project settings.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        logger.info("DatasetValidator initialized")

    def validate_all(self) -> ValidationReport:
        """
        Run complete validation on both datasets.

        Returns:
            ValidationReport with full results and verdict
        """
        logger.info("Starting dataset validation...")
        report = ValidationReport()

        # Validate each source dataset
        pv_counts = self._validate_source(
            source_name="PlantVillage",
            source_path=self.settings.plantvillage_path,
            class_names=self.settings.plantvillage_classes,
            report=report
        )

        md_counts = self._validate_source(
            source_name="Mendeley",
            source_path=self.settings.mendeley_path,
            class_names=self.settings.mendeley_classes,
            report=report
        )

        # Merge counts using unified class names
        self._merge_class_counts(
            pv_counts,
            md_counts,
            report
        )

        # Check class balance across unified classes
        self._check_class_balance(report)

        # Final verdict
        report.passed = len(report.errors) == 0
        self._log_report(report)

        return report

    def _validate_source(
        self,
        source_name: str,
        source_path: Path,
        class_names: List[str],
        report: ValidationReport
    ) -> Dict[str, int]:
        """
        Validate a single dataset source.

        Args:
            source_name: Human-readable name for logging
            source_path: Path to the dataset root directory
            class_names: Expected class folder names
            report: ValidationReport to append findings to

        Returns:
            Dictionary of class name to image count
        """
        logger.info(f"Validating {source_name} dataset...")
        counts = {}

        # Check source directory exists
        if not source_path.exists():
            error = (
                f"{source_name} directory not found: "
                f"{source_path}"
            )
            report.errors.append(error)
            logger.error(error)
            return counts

        logger.info(f"  Path: {source_path}")

        # Validate each class folder
        for class_name in class_names:
            class_path = source_path / class_name
            count = self._validate_class_folder(
                source_name=source_name,
                class_name=class_name,
                class_path=class_path,
                report=report
            )
            counts[class_name] = count

        total = sum(counts.values())
        logger.info(
            f"  {source_name} total: {total} images "
            f"across {len(class_names)} classes"
        )
        return counts

    def _validate_class_folder(
        self,
        source_name: str,
        class_name: str,
        class_path: Path,
        report: ValidationReport
    ) -> int:
        """
        Validate a single class folder.

        Checks existence, minimum image count, and optionally
        samples images for corruption check.

        Returns:
            Number of valid images found
        """
        # Check folder exists
        if not class_path.exists():
            msg = (
                f"{source_name}/{class_name} folder "
                f"not found: {class_path}"
            )
            report.missing_folders.append(str(class_path))
            report.errors.append(msg)
            logger.error(f"  ✗ Missing: {class_name}")
            return 0

        # Count images
        images = list_image_files(
            class_path,
            self.settings.valid_extensions
        )
        count = len(images)

        # Check minimum count
        min_required = self.settings.min_images_per_class
        if count < min_required:
            msg = (
                f"{source_name}/{class_name} has only "
                f"{count} images (minimum: {min_required})"
            )
            report.errors.append(msg)
            logger.error(f"  ✗ {class_name}: {count} images")
        else:
            logger.info(
                f"  ✓ {class_name}: {count} images"
            )

        # Sample 5 images for corruption check
        sample = images[:5]
        corrupted = [
            str(img) for img in sample
            if not is_valid_image(img)
        ]
        if corrupted:
            report.corrupted_files.extend(corrupted)
            msg = (
                f"{source_name}/{class_name}: "
                f"{len(corrupted)} corrupted files found"
            )
            report.warnings.append(msg)
            logger.warning(f"  ⚠ {msg}")

        return count

    def _merge_class_counts(
        self,
        pv_counts: Dict[str, int],
        md_counts: Dict[str, int],
        report: ValidationReport
    ) -> None:
        """
        Merge per-source counts into unified class counts.

        Maps source-specific folder names to unified class
        names using the class_mapping from config.
        """
        mapping = self.settings.class_mapping
        unified = {}

        for source_class, count in pv_counts.items():
            unified_name = mapping.get(source_class, source_class)
            unified[unified_name] = (
                unified.get(unified_name, 0) + count
            )

        for source_class, count in md_counts.items():
            unified_name = mapping.get(source_class, source_class)
            unified[unified_name] = (
                unified.get(unified_name, 0) + count
            )

        report.images_per_class = unified
        report.total_images = sum(unified.values())

    def _check_class_balance(
        self,
        report: ValidationReport
    ) -> None:
        """
        Check for significant class imbalance.

        Warns if the largest class has more than
        warn_imbalance_ratio times the smallest class.
        """
        counts = report.images_per_class
        if not counts:
            return

        max_count = max(counts.values())
        min_count = min(counts.values())

        if min_count == 0:
            return

        ratio = max_count / min_count
        threshold = self.settings.warn_imbalance_ratio

        if ratio > threshold:
            max_class = max(counts, key=counts.get)
            min_class = min(counts, key=counts.get)
            msg = (
                f"Class imbalance detected — ratio: {ratio:.2f}x "
                f"({max_class}: {max_count} vs "
                f"{min_class}: {min_count}). "
                f"Augmentation will compensate."
            )
            report.warnings.append(msg)
            logger.warning(f"  ⚠ {msg}")
        else:
            logger.info(
                f"  ✓ Class balance acceptable "
                f"(ratio: {ratio:.2f}x)"
            )

    def _log_report(self, report: ValidationReport) -> None:
        """Log the full validation report."""
        for line in report.summary().split("\n"):
            if "PASSED" in line:
                logger.info(line)
            elif "FAILED" in line or "✗" in line:
                logger.error(line)
            elif "⚠" in line:
                logger.warning(line)
            else:
                logger.info(line)