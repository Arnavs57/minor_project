"""
Color correction for underwater images.

Underwater images suffer from blue/green color dominance due to wavelength-dependent
light absorption (red light absorbed first, blue/green penetrates deeper).
These methods attempt to restore more natural color balance.
"""

import cv2
import numpy as np
from typing import Literal


def apply_gray_world(image: np.ndarray) -> np.ndarray:
    """
    Gray World color correction.
    
    Assumption: Average color in the scene should be gray.
    Scales each channel so that mean(R) = mean(G) = mean(B).
    Simple and effective for mild color casts.
    
    Args:
        image: BGR image (OpenCV format).
        
    Returns:
        Color-corrected BGR image.
    """
    # Compute mean of each channel
    mean_b = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_r = np.mean(image[:, :, 2])
    
    # Target: gray (equal means)
    gray_mean = (mean_b + mean_g + mean_r) / 3.0
    
    # Avoid division by zero
    mean_b = max(mean_b, 1e-6)
    mean_g = max(mean_g, 1e-6)
    mean_r = max(mean_r, 1e-6)
    
    # Scale each channel
    result = image.copy().astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] * (gray_mean / mean_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (gray_mean / mean_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (gray_mean / mean_r), 0, 255)
    
    return result.astype(np.uint8)


def apply_white_balance(
    image: np.ndarray,
    method: Literal["gray_world", "simple"] = "gray_world"
) -> np.ndarray:
    """
    White balance / color correction.
    
    Args:
        image: BGR image.
        method: "gray_world" or "simple" (percentile-based).
        
    Returns:
        White-balanced BGR image.
    """
    if method == "gray_world":
        return apply_gray_world(image)
    
    if method == "simple":
        # Simple white balance: scale channels to match max channel
        result = image.astype(np.float32)
        for c in range(3):
            channel = result[:, :, c]
            # Use 98th percentile to avoid outliers
            p_high = np.percentile(channel, 98)
            p_low = np.percentile(channel, 2)
            if p_high > p_low:
                result[:, :, c] = np.clip(
                    (channel - p_low) * (255.0 / (p_high - p_low)), 0, 255
                )
        return result.astype(np.uint8)
    
    raise ValueError(f"Unknown method: {method}")
