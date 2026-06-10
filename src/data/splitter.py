# =============================================================
# src/data/splitter.py
# Deterministic train/validation/test dataset splitting
# =============================================================
# Splits processed images into three sets with guaranteed
# reproducibility. Saves split manifests as CSV files so
# the exact same split can be reproduced on any machine.
# Uses stratified splitting to maintain class proportions.
# =============================================================

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.file_utils import (
    list_image_files,
    ensure_directory
)
from src.config.settings import Settings

logger = get_logger(__name__)


@dataclass
class SplitManifest:
    """
    Records which image belongs to which split.

    Saved as CSV files — one per split — so the exact
    same train/val/test division can always be reproduced.
    """
    train: List[Tuple[str, str]] = field(default_factory=list)
    val: List[Tuple[str, str]] = field(default_factory=list)
    test: List[Tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.train) + len(self.val) + len(self.test)
        lines = [
            "=" * 60,
            "DATASET SPLIT REPORT",
            "=" * 60,
            f"Total images : {total}",
            f"Train        : {len(self.train)} "
            f"({len(self.train)/total*100:.1f}%)",
            f"Validation   : {len(self.val)} "
            f"({len(self.val)/total*100:.1f}%)",
            f"Test         : {len(self.test)} "
            f"({len(self.test)/total*100:.1f}%)",
            "=" * 60,
        ]
        return "\n".join(lines)


class DatasetSplitter:
    """
    Splits processed images into train/validation/test sets.

    Key properties:
    - Deterministic: same seed always produces same split
    - Stratified: each split has same class proportions
    - Manifest-based: saves CSV records for reproducibility
    - Non-destructive: never moves or copies image files

    The splitter works on file paths only — it records
    which image goes where without touching the actual files.
    TensorFlow then reads images directly from their paths.

    Usage:
        settings = Settings()
        splitter = DatasetSplitter(settings)
        manifest = splitter.split()
    """

    def __init__(self, settings: Settings):
        """
        Initialize splitter with project settings.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.seed = settings.random_seed
        self.train_ratio = settings.train_ratio
        self.val_ratio = settings.val_ratio
        self.test_ratio = settings.test_ratio
        self.processed_dir = settings.processed_output
        self.splits_dir = settings.splits_output

        logger.info(
            f"DatasetSplitter initialized — "
            f"ratios: {self.train_ratio:.0%}/"
            f"{self.val_ratio:.0%}/"
            f"{self.test_ratio:.0%} "
            f"seed: {self.seed}"
        )

    def split(self) -> SplitManifest:
        """
        Perform stratified split across all classes.

        For each class:
        1. List all processed images
        2. Shuffle with fixed seed
        3. Split into train/val/test by ratio
        4. Add to manifest

        Returns:
            SplitManifest with train/val/test file lists
        """
        logger.info("Starting dataset splitting...")
        manifest = SplitManifest()

        unified_classes = self.settings.unified_classes

        for class_name in unified_classes:
            class_dir = self.processed_dir / class_name

            if not class_dir.exists():
                logger.warning(
                    f"Processed class folder not found: "
                    f"{class_dir} — skipping"
                )
                continue

            # Get all images for this class
            images = list_image_files(
                class_dir,
                self.settings.valid_extensions
            )

            if not images:
                logger.warning(
                    f"No images found in {class_name} — "
                    f"skipping"
                )
                continue

            logger.info(
                f"  Splitting {class_name}: "
                f"{len(images)} images"
            )

            # Deterministic shuffle — same seed, same order
            random.seed(self.seed)
            shuffled = images.copy()
            random.shuffle(shuffled)

            # Calculate split indices
            n = len(shuffled)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            # Split the list
            train_imgs = shuffled[:train_end]
            val_imgs = shuffled[train_end:val_end]
            test_imgs = shuffled[val_end:]

            # Add to manifest as (path, class) tuples
            manifest.train.extend(
                [(str(p), class_name) for p in train_imgs]
            )
            manifest.val.extend(
                [(str(p), class_name) for p in val_imgs]
            )
            manifest.test.extend(
                [(str(p), class_name) for p in test_imgs]
            )

            logger.info(
                f"    Train: {len(train_imgs)} | "
                f"Val: {len(val_imgs)} | "
                f"Test: {len(test_imgs)}"
            )

        # Save manifests to CSV files
        ensure_directory(self.splits_dir)
        self._save_manifest(manifest)

        # Log summary
        for line in manifest.summary().split("\n"):
            logger.info(line)

        return manifest

    def _save_manifest(self, manifest: SplitManifest) -> None:
        """
        Save split manifests as CSV files.

        Creates three files:
            data/processed/splits/train.csv
            data/processed/splits/val.csv
            data/processed/splits/test.csv

        Each CSV has two columns: image_path, class_name

        These files are committed to Git — guaranteeing
        the exact same split can always be reproduced.
        """
        splits = {
            "train": manifest.train,
            "val": manifest.val,
            "test": manifest.test
        }

        for split_name, records in splits.items():
            csv_path = self.splits_dir / f"{split_name}.csv"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "class_name"])
                writer.writerows(records)

            logger.info(
                f"  Saved {split_name} manifest: "
                f"{len(records)} records → {csv_path}"
            )

    def load_manifest(self) -> SplitManifest:
        """
        Load previously saved split manifests from CSV.

        Use this to reload splits without re-splitting —
        guarantees the same images are always in the same set.

        Returns:
            SplitManifest loaded from saved CSV files

        Raises:
            FileNotFoundError: If CSV files don't exist yet
        """
        logger.info("Loading split manifests from CSV...")
        manifest = SplitManifest()

        for split_name in ["train", "val", "test"]:
            csv_path = self.splits_dir / f"{split_name}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Split manifest not found: {csv_path}\n"
                    f"Run splitter.split() first."
                )

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                records = [
                    (row["image_path"], row["class_name"])
                    for row in reader
                ]

            if split_name == "train":
                manifest.train = records
            elif split_name == "val":
                manifest.val = records
            else:
                manifest.test = records

            logger.info(
                f"  Loaded {split_name}: "
                f"{len(records)} records"
            )

        return manifest

    def get_class_distribution(
        self,
        manifest: SplitManifest
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate class distribution across all splits.

        Returns per-split, per-class image counts.
        Useful for verifying stratification worked correctly.

        Returns:
            Nested dict: {split: {class: count}}
        """
        distribution = {
            "train": {},
            "val": {},
            "test": {}
        }

        for path, cls in manifest.train:
            distribution["train"][cls] = (
                distribution["train"].get(cls, 0) + 1
            )
        for path, cls in manifest.val:
            distribution["val"][cls] = (
                distribution["val"].get(cls, 0) + 1
            )
        for path, cls in manifest.test:
            distribution["test"][cls] = (
                distribution["test"].get(cls, 0) + 1
            )

        return distribution