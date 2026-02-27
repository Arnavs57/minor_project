"""
Contrast enhancement for underwater images using CLAHE.

CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local contrast
without over-amplifying noise. It works on small tiles and limits contrast
amplification, making it suitable for low-visibility underwater images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_lab: bool = True
) -> np.ndarray:
    """
    Apply CLAHE for contrast enhancement.
    
    When use_lab=True, CLAHE is applied only on the L (luminance) channel
    in LAB color space to avoid color distortion. This is recommended for
    underwater images.
    
    Args:
        image: BGR image (OpenCV format).
        clip_limit: Contrast limiting factor (1.0-4.0 typical). Higher = more contrast.
        tile_grid_size: Grid size for local histogram. Smaller = more local effect.
        use_lab: If True, apply on L channel in LAB (preserves color). Else on grayscale.
        
    Returns:
        Contrast-enhanced BGR image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if use_lab:
        # Convert to LAB: L = luminance, A/B = color
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = clahe.apply(gray)
        result = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    return result
