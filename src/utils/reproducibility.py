# =============================================================
# src/utils/reproducibility.py
# Deterministic seeding for full pipeline reproducibility
# =============================================================
# Sets ALL random number generators to the same seed value.
# This guarantees that every pipeline run produces identical
# results — same splits, same augmentation, same weight init.
# Critical for academic research and result verification.
# =============================================================

import os
import random
import numpy as np
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seeds(seed: int) -> None:
    """
    Set all random seeds across every library simultaneously.

    In machine learning, randomness appears in multiple places:
    - Dataset shuffling (Python random)
    - Weight initialization (NumPy)
    - Model operations (TensorFlow)
    - Hash-based operations (Python internals)

    Setting all of them to the same value guarantees that
    running this pipeline twice gives identical results.

    Args:
        seed: Integer seed value — loaded from config (default 42)

    Example:
        from src.utils.reproducibility import set_global_seeds
        set_global_seeds(42)
    """

    logger.info(f"Setting global random seed: {seed}")

    # ── Python built-in random ───────────────────────────────
    random.seed(seed)

    # ── Python hash seed (affects dict ordering etc.) ────────
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ── NumPy random ─────────────────────────────────────────
    np.random.seed(seed)

    # ── TensorFlow random ────────────────────────────────────
    tf.random.set_seed(seed)

    logger.info("All random seeds configured successfully")
    logger.info(f"  python random  : seed={seed}")
    logger.info(f"  PYTHONHASHSEED : {os.environ['PYTHONHASHSEED']}")
    logger.info(f"  numpy          : seed={seed}")
    logger.info(f"  tensorflow     : seed={seed}")


def get_seed_from_config(config: dict) -> int:
    """
    Safely extract seed value from configuration dictionary.

    Args:
        config: Full configuration dictionary from settings.py

    Returns:
        Integer seed value, defaults to 42 if not found

    Example:
        seed = get_seed_from_config(config)
        set_global_seeds(seed)
    """
    seed = config.get("project", {}).get("seed", 42)
    logger.debug(f"Seed loaded from config: {seed}")
    return seed