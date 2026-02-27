"""
Full preprocessing pipeline for underwater images.

Combines color correction, CLAHE, and dehazing in a configurable order.
Can process single images or entire directories.
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .color_correction import apply_gray_world
from .contrast_enhancement import apply_clahe
from .dehazing import apply_dark_channel_dehaze

# Add project root to path for utils
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import ensure_dir, get_image_paths, load_config


def preprocess_image(
    image: np.ndarray,
    color_correct: bool = True,
    clahe: bool = True,
    dehaze: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid: tuple = (8, 8),
    dehaze_omega: float = 0.95,
    dehaze_t0: float = 0.1,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single image.
    
    Order: Color correction -> CLAHE -> Dehazing
    (Color first to restore channels, then contrast, then reduce haze)
    
    Args:
        image: BGR image (numpy array from cv2.imread).
        color_correct: Apply gray world color correction.
        clahe: Apply CLAHE contrast enhancement.
        dehaze: Apply dark channel dehazing.
        clahe_clip_limit: CLAHE clip limit.
        clahe_grid: CLAHE tile grid size.
        dehaze_omega: Dehazing omega.
        dehaze_t0: Dehazing t0.
        
    Returns:
        Preprocessed BGR image.
    """
    result = image.copy()
    
    if color_correct:
        result = apply_gray_world(result)
    
    if clahe:
        result = apply_clahe(
            result,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_grid,
            use_lab=True,
        )
    
    if dehaze:
        result = apply_dark_channel_dehaze(
            result,
            omega=dehaze_omega,
            t0=dehaze_t0,
        )
    
    return result


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    config: Optional[dict] = None,
    recursive: bool = False,
) -> int:
    """
    Preprocess all images in a directory and save to output.
    
    Args:
        input_dir: Directory containing raw images.
        output_dir: Directory to save preprocessed images.
        config: Optional config dict (uses config.yaml if None).
        recursive: Search subdirectories.
        
    Returns:
        Number of images processed.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}
    
    prep_config = config.get("preprocessing", {})
    clahe_cfg = prep_config.get("clahe", {})
    dehaze_cfg = prep_config.get("dehazing", {})
    
    ensure_dir(output_dir)
    paths = get_image_paths(input_dir, recursive=recursive)
    
    if not paths:
        print(f"No images found in {input_dir}")
        return 0
    
    for path in tqdm(paths, desc="Preprocessing"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        
        out_img = preprocess_image(
            img,
            color_correct=prep_config.get("color_correction", {}).get("enabled", True),
            clahe=clahe_cfg.get("enabled", True),
            dehaze=dehaze_cfg.get("enabled", True),
            clahe_clip_limit=clahe_cfg.get("clip_limit", 2.0),
            clahe_grid=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
            dehaze_omega=dehaze_cfg.get("omega", 0.95),
            dehaze_t0=dehaze_cfg.get("t0", 0.1),
        )
        
        out_path = Path(output_dir) / path.name
        cv2.imwrite(str(out_path), out_img)
    
    return len(paths)


def main():
    """CLI entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess underwater images (color correction, CLAHE, dehazing)"
    )
    parser.add_argument(
        "--input", "-i",
        default="dataset/raw",
        help="Input directory with raw images",
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset/processed",
        help="Output directory for preprocessed images",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process subdirectories recursively",
    )
    args = parser.parse_args()
    
    count = preprocess_directory(
        args.input,
        args.output,
        recursive=args.recursive,
    )
    print(f"Processed {count} images -> {args.output}")


if __name__ == "__main__":
    main()
