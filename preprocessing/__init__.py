"""
Underwater image preprocessing module.

Provides enhancement techniques to address:
- Blue/green color dominance (color correction)
- Low contrast (CLAHE)
- Haze and scattering (dehazing)
"""

from .color_correction import apply_gray_world, apply_white_balance
from .contrast_enhancement import apply_clahe
from .dehazing import apply_dark_channel_dehaze
from .pipeline import preprocess_image, preprocess_directory
from .enhancement import (
    color_correction,
    color_correction_gray_world,
    color_correction_white_balance_percentile,
    color_correction_histogram_equalization,
    contrast_enhancement,
    contrast_enhancement_clahe,
    contrast_enhancement_histogram,
    enhance_underwater,
)

__all__ = [
    "apply_gray_world",
    "apply_white_balance",
    "apply_clahe",
    "apply_dark_channel_dehaze",
    "preprocess_image",
    "preprocess_directory",
    "color_correction",
    "color_correction_gray_world",
    "color_correction_white_balance_percentile",
    "color_correction_histogram_equalization",
    "contrast_enhancement",
    "contrast_enhancement_clahe",
    "contrast_enhancement_histogram",
    "enhance_underwater",
]
