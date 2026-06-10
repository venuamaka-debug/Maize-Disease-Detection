# =============================================================
# src/data/pipeline.py
# Pipeline orchestrator — coordinates all data modules
# =============================================================
# The conductor of the orchestra. Calls each module in the
# correct order, passes results between them, and produces
# a complete, ready-to-train TensorFlow dataset.
# This is the only file that knows the full pipeline order.
# =============================================================

import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seeds
from src.utils.file_utils import ensure_directory
from src.config.settings import Settings
from src.data.validator import DatasetValidator
from src.data.preprocessor import ImagePreprocessor
from src.data.augmentor import DataAugmentor
from src.data.splitter import DatasetSplitter, SplitManifest

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """
    Complete result of running the full pipeline.

    Contains TensorFlow datasets ready for model training
    plus all reports and statistics from each stage.
    """
    train_dataset: Optional[tf.data.Dataset] = None
    val_dataset: Optional[tf.data.Dataset] = None
    test_dataset: Optional[tf.data.Dataset] = None
    manifest: Optional[SplitManifest] = None
    num_classes: int = 4
    class_names: list = None
    image_size: tuple = (224, 224)
    batch_size: int = 32


class DataPipeline:
    """
    Orchestrates the complete data preprocessing pipeline.

    Execution order:
    1. Set random seeds for reproducibility
    2. Validate raw datasets
    3. Preprocess images (resize, normalize)
    4. Split into train/val/test
    5. Build TensorFlow datasets
    6. Apply augmentation to training set only

    Usage:
        settings = Settings()
        pipeline = DataPipeline(settings)

        # Full pipeline — use first time
        result = pipeline.run_full()

        # Quick reload — use after first run
        result = pipeline.run_from_manifests()
    """

    def __init__(self, settings: Settings):
        """
        Initialize pipeline with all required modules.

        Args:
            settings: Loaded Settings instance
        """
        self.settings = settings

        # Initialize all pipeline modules
        self.validator   = DatasetValidator(settings)
        self.preprocessor = ImagePreprocessor(settings)
        self.augmentor   = DataAugmentor(settings)
        self.splitter    = DatasetSplitter(settings)

        logger.info("DataPipeline initialized")
        logger.info(
            f"  Image size  : {settings.image_size}"
        )
        logger.info(
            f"  Batch size  : {settings.batch_size}"
        )
        logger.info(
            f"  Classes     : {settings.unified_classes}"
        )
        logger.info(
            f"  Split ratios: "
            f"{settings.train_ratio:.0%}/"
            f"{settings.val_ratio:.0%}/"
            f"{settings.test_ratio:.0%}"
        )

    def run_full(self) -> PipelineResult:
        """
        Run the complete pipeline from raw data to datasets.

        Use this on first run or when raw data changes.
        Validates → Preprocesses → Splits → Builds datasets.

        Returns:
            PipelineResult with ready-to-use TF datasets

        Raises:
            RuntimeError: If validation fails
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL DATA PIPELINE")
        logger.info("=" * 60)

        # ── Step 1: Reproducibility ──────────────────────────
        logger.info("Step 1/5: Setting random seeds...")
        set_global_seeds(self.settings.seed)

        # ── Step 2: Validation ───────────────────────────────
        logger.info("Step 2/5: Validating raw datasets...")
        validation_report = self.validator.validate_all()

        if not validation_report.passed:
            raise RuntimeError(
                "Dataset validation failed. "
                "Check logs for details. "
                "Fix errors before proceeding."
            )
        logger.info("  ✓ Validation passed")

        # ── Step 3: Preprocessing ────────────────────────────
        logger.info("Step 3/5: Preprocessing images...")
        logger.info(
            "  This may take several minutes "
            "for 17,000+ images..."
        )
        preprocessing_report = (
            self.preprocessor.preprocess_all()
        )
        logger.info(
            f"  ✓ Preprocessed "
            f"{preprocessing_report.total_processed} images"
        )

        # ── Step 4: Splitting ────────────────────────────────
        logger.info("Step 4/5: Splitting dataset...")
        manifest = self.splitter.split()
        logger.info(
            f"  ✓ Split complete — "
            f"Train: {len(manifest.train)} | "
            f"Val: {len(manifest.val)} | "
            f"Test: {len(manifest.test)}"
        )

        # ── Step 5: Build TF Datasets ────────────────────────
        logger.info("Step 5/5: Building TensorFlow datasets...")
        result = self._build_tf_datasets(manifest)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return result

    def run_from_manifests(self) -> PipelineResult:
        """
        Reload datasets from saved split manifests.

        Use this after the first run to skip preprocessing.
        Much faster — just loads paths and builds TF datasets.

        Returns:
            PipelineResult with ready-to-use TF datasets
        """
        logger.info("Loading pipeline from saved manifests...")
        set_global_seeds(self.settings.seed)

        manifest = self.splitter.load_manifest()
        result = self._build_tf_datasets(manifest)

        logger.info(
            f"Pipeline loaded — "
            f"Train: {len(manifest.train)} | "
            f"Val: {len(manifest.val)} | "
            f"Test: {len(manifest.test)}"
        )
        return result

    def _build_tf_datasets(
        self,
        manifest: SplitManifest
    ) -> PipelineResult:
        """
        Convert split manifests into TensorFlow datasets.

        Builds three tf.data.Dataset objects:
        - train_dataset: augmented, shuffled, batched
        - val_dataset: normalized only, batched
        - test_dataset: normalized only, batched

        Args:
            manifest: SplitManifest with file path lists

        Returns:
            PipelineResult with all three datasets
        """
        class_names = self.settings.unified_classes
        image_size = self.settings.image_size
        batch_size = self.settings.batch_size

        # Build augmentation layer for training only
        augmentation_layer = (
            self.augmentor.build_augmentation_layer()
        )

        # ── Training Dataset ─────────────────────────────────
        logger.info("  Building training dataset...")
        train_ds = self._build_dataset(
            records=manifest.train,
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            augmentation=augmentation_layer,
            shuffle=True
        )

        # ── Validation Dataset ───────────────────────────────
        logger.info("  Building validation dataset...")
        val_ds = self._build_dataset(
            records=manifest.val,
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            augmentation=None,
            shuffle=False
        )

        # ── Test Dataset ─────────────────────────────────────
        logger.info("  Building test dataset...")
        test_ds = self._build_dataset(
            records=manifest.test,
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            augmentation=None,
            shuffle=False
        )

        return PipelineResult(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            manifest=manifest,
            num_classes=len(class_names),
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size
        )

    def _build_dataset(
        self,
        records: list,
        class_names: list,
        image_size: tuple,
        batch_size: int,
        augmentation,
        shuffle: bool
    ) -> tf.data.Dataset:
        """
        Build a single tf.data.Dataset from manifest records.

        Args:
            records: List of (image_path, class_name) tuples
            class_names: Ordered list of unified class names
            image_size: Target image dimensions (H, W)
            batch_size: Number of images per batch
            augmentation: Augmentation layer or None
            shuffle: Whether to shuffle the dataset

        Returns:
            Optimized tf.data.Dataset
        """
        # Separate paths and labels
        paths = [r[0] for r in records]
        labels = [
            class_names.index(r[1]) for r in records
        ]

        # Create TensorFlow dataset from paths and labels
        path_ds = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        # Map: load and preprocess each image
        def load_and_preprocess(path, label):
            image = self._load_image(path, image_size)
            return image, label

        dataset = path_ds.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle training data
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.settings.shuffle_buffer,
                seed=self.settings.seed,
                reshuffle_each_iteration=True
            )

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Apply augmentation to training only
        if augmentation is not None:
            dataset = dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Prefetch for performance
        # Loads next batch while GPU/CPU processes current
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _load_image(
        self,
        path: str,
        image_size: tuple
    ) -> tf.Tensor:
        """
        Load and normalize a single image as TF tensor.

        Steps:
        1. Read file from disk
        2. Decode JPEG
        3. Resize to target dimensions
        4. Normalize pixels from [0,255] to [0.0,1.0]

        Args:
            path: File path string
            image_size: Target (height, width)

        Returns:
            Normalized float32 tensor of shape
            (height, width, 3)
        """
        # Read raw bytes from disk
        raw = tf.io.read_file(path)

        # Decode image — handles JPEG and PNG
        image = tf.image.decode_image(
            raw,
            channels=3,
            expand_animations=False
        )

        # Resize to target dimensions
        image = tf.image.resize(image, image_size)

        # Normalize pixels to [0.0, 1.0]
        if self.settings.normalize:
            image = tf.cast(image, tf.float32) / 255.0

        return image