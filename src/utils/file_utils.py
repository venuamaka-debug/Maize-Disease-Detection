# =============================================================
# src/utils/file_utils.py
# File system operations for the entire pipeline
# =============================================================
# Handles all path building, directory creation, and file
# listing operations. Every other module uses these utilities
# instead of writing their own path logic — keeping file
# operations consistent, safe, and platform independent.
# =============================================================

import os
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    This works regardless of where the script is called from
    by navigating up from this file's location.

    Returns:
        Path object pointing to project root

    Example:
        root = get_project_root()
        # Returns: /home/victor-e/maize-disease-detection
    """
    # This file is at src/utils/file_utils.py
    # So project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(relative_path: str) -> Path:
    """
    Convert a relative path from config into an absolute path.

    All paths in the config file are relative to project root.
    This function converts them to absolute paths that work
    regardless of which directory the script is run from.

    Args:
        relative_path: Path string from config file
                      e.g. "data/raw/plantvillage/data"

    Returns:
        Absolute Path object

    Example:
        path = resolve_path("data/raw/plantvillage/data")
        # Returns: /home/victor-e/maize-disease-detection/data/raw/plantvillage/data
    """
    root = get_project_root()
    resolved = root / relative_path
    logger.debug(f"Resolved path: {relative_path} → {resolved}")
    return resolved


def ensure_directory(path) -> Path:
    """
    Create a directory and all parent directories if needed.

    Safe to call even if directory already exists — it will
    not raise an error or overwrite existing content.

    Args:
        path: Path string or Path object to create

    Returns:
        Path object of the created directory

    Example:
        ensure_directory("data/processed/splits")
        # Creates the full path if it doesn't exist
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {dir_path}")
    return dir_path


def list_image_files(
    directory,
    valid_extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    List all valid image files in a directory.

    Recursively searches the directory and returns only files
    with valid image extensions. Skips hidden files, system
    files, and any non-image formats silently.

    Args:
        directory: Path to search for images
        valid_extensions: List of accepted extensions
                         Defaults to common image formats

    Returns:
        Sorted list of Path objects for all valid images

    Example:
        images = list_image_files("data/raw/plantvillage/data/Blight")
        print(f"Found {len(images)} images")
    """
    if valid_extensions is None:
        valid_extensions = [
            ".jpg", ".jpeg", ".png",
            ".JPG", ".JPEG", ".PNG"
        ]

    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return []

    image_files = []
    for ext in valid_extensions:
        # rglob searches recursively through all subdirectories
        image_files.extend(directory.rglob(f"*{ext}"))

    # Sort for consistent ordering across all runs
    image_files = sorted(set(image_files))

    logger.debug(
        f"Found {len(image_files)} images in {directory.name}"
    )
    return image_files


def count_images_per_class(
    source_dir,
    class_names: List[str],
    valid_extensions: Optional[List[str]] = None
) -> dict:
    """
    Count images in each class folder within a directory.

    Args:
        source_dir: Parent directory containing class folders
        class_names: List of expected class folder names
        valid_extensions: Accepted image file extensions

    Returns:
        Dictionary mapping class name to image count
        e.g. {"Blight": 1146, "Common_Rust": 1306, ...}

    Example:
        counts = count_images_per_class(
            "data/raw/plantvillage/data",
            ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
        )
    """
    counts = {}
    source_dir = Path(source_dir)

    for class_name in class_names:
        class_dir = source_dir / class_name
        images = list_image_files(class_dir, valid_extensions)
        counts[class_name] = len(images)
        logger.debug(f"  {class_name}: {len(images)} images")

    return counts


def is_valid_image(file_path) -> bool:
    """
    Check if a file is a valid, readable image.

    Attempts to open the file to verify it is not corrupted.
    Uses Pillow for validation without loading full image
    into memory — just reads the header.

    Args:
        file_path: Path to the image file to validate

    Returns:
        True if valid and readable, False if corrupted

    Example:
        if is_valid_image("data/raw/plantvillage/Blight/img.jpg"):
            process(img)
        else:
            logger.warning("Skipping corrupted image")
    """
    from PIL import Image

    try:
        with Image.open(file_path) as img:
            # verify() checks file integrity without
            # loading the full pixel data into memory
            img.verify()
        return True
    except Exception as e:
        logger.warning(
            f"Invalid image file: {file_path} — {str(e)}"
        )
        return False


def get_directory_size(directory) -> str:
    """
    Calculate total size of a directory in human-readable format.

    Useful for logging how much data was processed.

    Args:
        directory: Path to directory to measure

    Returns:
        Human-readable size string e.g. "1.23 GB"

    Example:
        size = get_directory_size("data/raw")
        logger.info(f"Raw data size: {size}")
    """
    directory = Path(directory)
    total_bytes = sum(
        f.stat().st_size
        for f in directory.rglob("*")
        if f.is_file()
    )

    # Convert to appropriate unit
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total_bytes < 1024:
            return f"{total_bytes:.2f} {unit}"
        total_bytes /= 1024

    return f"{total_bytes:.2f} TB"