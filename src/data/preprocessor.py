# =============================================================
# src/data/preprocessor.py
# Image preprocessing — resize, normalize, standardize
# =============================================================
# Transforms raw images into a standardized format ready
# for model input. Handles both datasets uniformly, maps
# source class names to unified labels, and saves processed
# images organized by unified class name.
# =============================================================

import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.utils.logger import get_logger
from src.utils.file_utils import (
    list_image_files,
    ensure_directory,
    is_valid_image
)
from src.config.settings import Settings

logger = get_logger(__name__)


@dataclass
class PreprocessingReport:
    """
    Structured report produced after preprocessing completes.
    """
    total_processed: int = 0
    total_skipped: int = 0
    total_corrupted: int = 0
    processed_per_class: Dict[str, int] = field(
        default_factory=dict
    )
    skipped_files: List[str] = field(default_factory=list)
    output_directory: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PREPROCESSING REPORT",
            "=" * 60,
            f"Total processed  : {self.total_processed}",
            f"Total skipped    : {self.total_skipped}",
            f"Total corrupted  : {self.total_corrupted}",
            f"Output directory : {self.output_directory}",
            "-" * 60,
            "Processed per class:",
        ]
        for cls, count in self.processed_per_class.items():
            lines.append(f"  {cls:<35} {count:>6} images")
        lines.append("=" * 60)
        return "\n".join(lines)


class ImagePreprocessor:
    """
    Preprocesses raw images into standardized format.

    For each image:
    1. Validates the file is readable
    2. Opens and converts to RGB
    3. Resizes to target dimensions (224x224)
    4. Saves to output directory organized by unified class

    The output directory structure matches what TensorFlow's
    ImageDataGenerator expects — one folder per class.

    Usage:
        settings = Settings()
        preprocessor = ImagePreprocessor(settings)
        report = preprocessor.preprocess_all()
    """

    def __init__(self, settings: Settings):
        """
        Initialize preprocessor with project settings.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings
        self.target_size = settings.image_size
        self.class_mapping = settings.class_mapping
        self.output_dir = settings.processed_output
        logger.info(
            f"ImagePreprocessor initialized — "
            f"target size: {self.target_size}"
        )

    def preprocess_all(self) -> PreprocessingReport:
        """
        Preprocess all images from both datasets.

        Processes PlantVillage first, then Mendeley.
        Both are saved into the same unified output structure.

        Returns:
            PreprocessingReport with full statistics
        """
        logger.info("Starting image preprocessing...")
        report = PreprocessingReport()
        report.output_directory = str(self.output_dir)

        # Create output directories for each unified class
        self._create_output_structure()

        # Process PlantVillage dataset
        logger.info("Processing PlantVillage dataset...")
        self._preprocess_source(
            source_path=self.settings.plantvillage_path,
            class_names=self.settings.plantvillage_classes,
            source_name="PlantVillage",
            report=report
        )

        # Process Mendeley dataset
        logger.info("Processing Mendeley dataset...")
        self._preprocess_source(
            source_path=self.settings.mendeley_path,
            class_names=self.settings.mendeley_classes,
            source_name="Mendeley",
            report=report
        )

        # Log final report
        for line in report.summary().split("\n"):
            logger.info(line)

        return report

    def _create_output_structure(self) -> None:
        """
        Create one output folder per unified class.

        Output structure:
            data/processed/
                Northern_Corn_Leaf_Blight/
                Common_Rust/
                Gray_Leaf_Spot/
                Healthy/
        """
        logger.info("Creating output directory structure...")
        for unified_class in self.settings.unified_classes:
            class_dir = self.output_dir / unified_class
            ensure_directory(class_dir)
            logger.info(f"  Created: {class_dir}")

    def _preprocess_source(
        self,
        source_path: Path,
        class_names: List[str],
        source_name: str,
        report: PreprocessingReport
    ) -> None:
        """
        Preprocess all classes from one dataset source.

        Args:
            source_path: Root path of the dataset
            class_names: List of class folder names
            source_name: Name for logging (PlantVillage/Mendeley)
            report: Report to update with statistics
        """
        for class_name in class_names:
            class_path = source_path / class_name
            unified_name = self.class_mapping.get(
                class_name, class_name
            )

            if not class_path.exists():
                logger.warning(
                    f"Skipping missing folder: {class_path}"
                )
                continue

            logger.info(
                f"  Processing {source_name}/{class_name} "
                f"→ {unified_name}"
            )

            processed, skipped = self._preprocess_class(
                class_path=class_path,
                unified_name=unified_name,
                source_name=source_name
            )

            # Update report counts
            current = report.processed_per_class.get(
                unified_name, 0
            )
            report.processed_per_class[unified_name] = (
                current + processed
            )
            report.total_processed += processed
            report.total_skipped += skipped

            logger.info(
                f"    ✓ {processed} processed, "
                f"{skipped} skipped"
            )

    def _preprocess_class(
        self,
        class_path: Path,
        unified_name: str,
        source_name: str
    ) -> Tuple[int, int]:
        """
        Preprocess all images in one class folder.

        Args:
            class_path: Path to the class folder
            unified_name: Unified class name for output
            source_name: Dataset source name for unique naming

        Returns:
            Tuple of (processed_count, skipped_count)
        """
        images = list_image_files(
            class_path,
            self.settings.valid_extensions
        )

        output_class_dir = self.output_dir / unified_name
        processed = 0
        skipped = 0

        for img_path in images:
            # Generate unique output filename
            # prefix with source to avoid name collisions
            output_name = f"{source_name}_{img_path.name}"
            output_path = output_class_dir / output_name

            # Skip if already processed
            if output_path.exists():
                skipped += 1
                continue

            # Preprocess and save
            success = self._process_single_image(
                input_path=img_path,
                output_path=output_path
            )

            if success:
                processed += 1
            else:
                skipped += 1

        return processed, skipped

    def _process_single_image(
        self,
        input_path: Path,
        output_path: Path
    ) -> bool:
        """
        Process a single image file.

        Steps:
        1. Validate file is readable
        2. Open with Pillow
        3. Convert to RGB (handles RGBA, grayscale edge cases)
        4. Resize to target size using high-quality resampling
        5. Save as JPEG to output path

        Args:
            input_path: Source image path
            output_path: Destination path for processed image

        Returns:
            True if successful, False if failed
        """
        try:
            with Image.open(input_path) as img:
                # Convert to RGB — handles PNG with alpha
                # channel, grayscale images, etc.
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize using LANCZOS for best quality
                # LANCZOS is the gold standard for downsampling
                img_resized = img.resize(
                    self.target_size,
                    Image.Resampling.LANCZOS
                )

                # Save as JPEG with high quality
                img_resized.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                    optimize=True
                )

            return True

        except Exception as e:
            logger.warning(
                f"Failed to process {input_path.name}: {e}"
            )
            return False