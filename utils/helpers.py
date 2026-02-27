"""
Common utility functions for the project.
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml


# Supported image extensions for dataset loading
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (relative to project root).
        
    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: str) -> Path:
    """
    Create directory if it does not exist.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_image_paths(
    directory: str,
    extensions: Optional[set] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Get all image file paths from a directory.
    
    Args:
        directory: Root directory to search.
        extensions: Set of valid extensions (default: common image formats).
        recursive: If True, search subdirectories.
        
    Returns:
        List of Path objects for image files.
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    
    root = Path(directory)
    if not root.exists():
        return []
    
    paths = []
    pattern = "**/*" if recursive else "*"
    
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p)
    
    return sorted(paths)
