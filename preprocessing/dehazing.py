"""
Dehazing for underwater images using Dark Channel Prior.

Underwater images often have haze/scattering similar to atmospheric haze.
The Dark Channel Prior assumes that in most patches, at least one color
channel has very low intensity. This helps estimate transmission and
atmospheric light for dehazing.
"""

import cv2
import numpy as np
from typing import Optional


def _min_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Minimum filter over local patch (for dark channel)."""
    return cv2.erode(image, np.ones((size, size)))


def _dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute dark channel: min over patch of min over RGB channels.
    """
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    return _min_filter(min_channel, patch_size)


def apply_dark_channel_dehaze(
    image: np.ndarray,
    omega: float = 0.95,
    t0: float = 0.1,
    patch_size: int = 15,
    radius: int = 60
) -> np.ndarray:
    """
    Dehaze using Dark Channel Prior (He et al.).
    
    Underwater: similar scattering model. omega controls amount of haze removed.
    t0 is minimum transmission to avoid division issues.
    
    Args:
        image: BGR image (uint8).
        omega: Amount of haze to keep (0.9-1.0). Lower = more aggressive dehazing.
        t0: Minimum transmission (prevents over-amplification).
        patch_size: Patch size for dark channel (odd, e.g. 15).
        radius: Radius for guided filter refinement (optional, not used in basic version).
        
    Returns:
        Dehazed BGR image.
    """
    # Ensure patch_size is odd
    patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
    
    img = image.astype(np.float64) / 255.0
    
    # 1. Dark channel
    dark = _dark_channel((img * 255).astype(np.uint8), patch_size)
    dark = dark.astype(np.float64) / 255.0
    
    # 2. Estimate atmospheric light (top 0.1% brightest in dark channel)
    h, w = image.shape[:2]
    flat_dark = dark.ravel()
    num_pixels = h * w
    num_top = max(int(num_pixels * 0.001), 1)
    indices = np.argpartition(flat_dark, -num_top)[-num_top:]
    
    # Among these, pick the one with highest intensity in original image
    brightest = 0
    atm_light = np.zeros(3)
    for i in indices:
        row, col = i // w, i % w
        intensity = np.mean(img[row, col, :])
        if intensity > brightest:
            brightest = intensity
            atm_light = img[row, col, :].copy()
    
    # 3. Transmission estimate: t(x) = 1 - omega * dark_channel(I/A)
    # Avoid division by zero
    atm_light = np.maximum(atm_light, 1e-6)
    normalized = np.zeros_like(img)
    for c in range(3):
        normalized[:, :, c] = img[:, :, c] / atm_light[c]
    
    dark_norm = _dark_channel((np.clip(normalized, 0, 1) * 255).astype(np.uint8), patch_size)
    dark_norm = dark_norm.astype(np.float64) / 255.0
    
    transmission = 1.0 - omega * dark_norm
    transmission = np.maximum(transmission, t0)
    
    # 4. Recover scene: J = (I - A) / t + A
    result = np.zeros_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - atm_light[c]) / transmission + atm_light[c]
    
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)
