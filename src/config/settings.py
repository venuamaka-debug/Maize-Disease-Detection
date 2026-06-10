# =============================================================
# src/config/settings.py
# Configuration loader — reads YAML and validates settings
# =============================================================
# This is the single entry point for all configuration.
# Every module imports settings from here — never reads
# the YAML file directly. This adds validation, type safety,
# and a single place to catch configuration mistakes early.
# =============================================================

import yaml
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.file_utils import get_project_root

logger = get_logger(__name__)


class Settings:
    """
    Configuration manager for the entire pipeline.

    Loads the YAML config file, validates required fields,
    and provides clean access to all settings throughout
    the codebase.

    Usage:
        settings = Settings()
        size = settings.image_size        # (224, 224)
        seed = settings.seed              # 42
        batch = settings.batch_size       # 32

    Or with a custom config path:
        settings = Settings("configs/my_config.yaml")
    """

    def __init__(
        self,
        config_path: str = "configs/preprocessing_config.yaml"
    ):
        """
        Load and validate configuration from YAML file.

        Args:
            config_path: Relative path to the YAML config file
                        Defaults to the standard config location
        """
        self._config_path = get_project_root() / config_path
        self._config = self._load_config()
        self._validate_config()
        logger.info(f"Configuration loaded: {config_path}")
        logger.info(
            f"Project: {self.project_name} v{self.version}"
        )

    # ── Config Loading ───────────────────────────────────────

    def _load_config(self) -> dict:
        """
        Read and parse the YAML configuration file.

        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If YAML syntax is invalid
        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._config_path}\n"
                f"Make sure you are running from the project "
                f"root directory."
            )

        logger.debug(f"Loading config from: {self._config_path}")

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(
                f"Config file is empty: {self._config_path}"
            )

        logger.debug("YAML parsed successfully")
        return config

    def _validate_config(self) -> None:
        """
        Validate that all required configuration sections exist.

        Catches missing sections early — before they cause
        cryptic errors deep inside the pipeline.

        Raises:
            KeyError: If a required section is missing
        """
        required_sections = [
            "project",
            "data",
            "image",
            "splitting",
            "augmentation",
            "preprocessing",
            "validation",
            "logging"
        ]

        missing = [
            section for section in required_sections
            if section not in self._config
        ]

        if missing:
            raise KeyError(
                f"Missing required config sections: {missing}\n"
                f"Check your preprocessing_config.yaml file."
            )

        # Validate split ratios sum to 1.0
        splitting = self._config["splitting"]
        total = (
            splitting["train_ratio"] +
            splitting["val_ratio"] +
            splitting["test_ratio"]
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f}\n"
                f"Check train_ratio, val_ratio, test_ratio "
                f"in config."
            )

        logger.debug("Configuration validation passed")

    # ── Project Settings ─────────────────────────────────────

    @property
    def project_name(self) -> str:
        return self._config["project"]["name"]

    @property
    def version(self) -> str:
        return self._config["project"]["version"]

    @property
    def author(self) -> str:
        return self._config["project"]["author"]

    @property
    def seed(self) -> int:
        return self._config["project"]["seed"]

    # ── Image Settings ───────────────────────────────────────

    @property
    def image_size(self) -> tuple:
        size = self._config["image"]["target_size"]
        return tuple(size)

    @property
    def channels(self) -> int:
        return self._config["image"]["channels"]

    @property
    def color_mode(self) -> str:
        return self._config["image"]["color_mode"]

    @property
    def valid_extensions(self) -> list:
        return self._config["image"]["valid_extensions"]

    # ── Data Path Settings ───────────────────────────────────

    @property
    def plantvillage_path(self) -> Path:
        raw = self._config["data"]["sources"]["plantvillage"]
        return get_project_root() / raw

    @property
    def mendeley_path(self) -> Path:
        raw = self._config["data"]["sources"]["mendeley"]
        return get_project_root() / raw

    @property
    def processed_output(self) -> Path:
        raw = self._config["data"]["processed_output"]
        return get_project_root() / raw

    @property
    def splits_output(self) -> Path:
        raw = self._config["data"]["splits_output"]
        return get_project_root() / raw

    @property
    def plantvillage_classes(self) -> list:
        return self._config["data"]["classes"]["plantvillage"]

    @property
    def mendeley_classes(self) -> list:
        return self._config["data"]["classes"]["mendeley"]

    @property
    def class_mapping(self) -> dict:
        return self._config["data"]["class_mapping"]

    @property
    def unified_classes(self) -> list:
        """Return the 4 unified class names used in training."""
        return list(dict.fromkeys(
            self.class_mapping.values()
        ))

    # ── Splitting Settings ───────────────────────────────────

    @property
    def train_ratio(self) -> float:
        return self._config["splitting"]["train_ratio"]

    @property
    def val_ratio(self) -> float:
        return self._config["splitting"]["val_ratio"]

    @property
    def test_ratio(self) -> float:
        return self._config["splitting"]["test_ratio"]

    @property
    def random_seed(self) -> int:
        return self._config["splitting"]["random_seed"]

    @property
    def stratified(self) -> bool:
        return self._config["splitting"]["stratified"]

    # ── Preprocessing Settings ───────────────────────────────

    @property
    def batch_size(self) -> int:
        return self._config["preprocessing"]["batch_size"]

    @property
    def normalize(self) -> bool:
        return self._config["preprocessing"]["normalize"]

    @property
    def shuffle_buffer(self) -> int:
        return self._config["preprocessing"]["shuffle_buffer"]

    # ── Augmentation Settings ────────────────────────────────

    @property
    def augmentation_enabled(self) -> bool:
        return self._config["augmentation"]["enabled"]

    @property
    def augmentation_config(self) -> dict:
        return self._config["augmentation"]

    # ── Validation Thresholds ────────────────────────────────

    @property
    def min_images_per_class(self) -> int:
        return self._config["validation"]["min_images_per_class"]

    @property
    def max_corrupted_ratio(self) -> float:
        return self._config["validation"]["max_corrupted_ratio"]

    @property
    def warn_imbalance_ratio(self) -> float:
        return self._config["validation"]["warn_imbalance_ratio"]

    # ── Logging Settings ─────────────────────────────────────

    @property
    def logging_config(self) -> dict:
        return self._config["logging"]

    # ── Raw Config Access ────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get any config value by dot-notation key.

        Args:
            key: Dot-separated key e.g. "image.target_size"
            default: Value to return if key not found

        Example:
            size = settings.get("image.target_size")
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"project='{self.project_name}', "
            f"version='{self.version}', "
            f"image_size={self.image_size}, "
            f"seed={self.seed}"
            f")"
        )