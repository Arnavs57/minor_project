"""
Underwater image color correction and contrast enhancement using OpenCV.

Provides a unified API for enhancing underwater images that suffer from
blue/green color cast and low contrast. All operations use OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Literal


# -----------------------------------------------------------------------------
# Color correction
# -----------------------------------------------------------------------------


def color_correction_gray_world(image: np.ndarray) -> np.ndarray:
    """
    Gray World color correction (OpenCV / NumPy).

    Assumption: average scene color should be neutral gray. We scale each BGR
    channel so that their means are equal, reducing blue/green dominance
    typical in underwater images.

    Args:
        image: BGR image (uint8), OpenCV format.

    Returns:
        Color-corrected BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(image)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (mean_b + mean_g + mean_r) / 3.0

    mean_b = max(mean_b, 1e-6)
    mean_g = max(mean_g, 1e-6)
    mean_r = max(mean_r, 1e-6)

    b = np.clip(b.astype(np.float32) * (gray_mean / mean_b), 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * (gray_mean / mean_g), 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.float32) * (gray_mean / mean_r), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])


def color_correction_white_balance_percentile(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """
    Percentile-based white balance.

    Scales each channel so that (low_percentile, high_percentile) map toward
    (0, 255), reducing color cast while limiting impact of outliers.

    Args:
        image: BGR image (uint8).
        low_percentile: Lower percentile for scaling.
        high_percentile: Upper percentile for scaling.

    Returns:
        White-balanced BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    result = image.astype(np.float32)
    for c in range(3):
        ch = result[:, :, c]
        p_low = np.percentile(ch, low_percentile)
        p_high = np.percentile(ch, high_percentile)
        if p_high > p_low:
            result[:, :, c] = np.clip((ch - p_low) * (255.0 / (p_high - p_low)), 0, 255)
        else:
            result[:, :, c] = ch

    return result.astype(np.uint8)


def color_correction_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Per-channel histogram equalization in BGR.

    Improves contrast per channel and can reduce color cast; may amplify noise.
    For smoother results, prefer CLAHE on luminance (see contrast_enhancement_clahe).

    Args:
        image: BGR image (uint8).

    Returns:
        Color-corrected BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    channels = list(cv2.split(image))
    for i in range(3):
        channels[i] = cv2.equalizeHist(channels[i])
    return cv2.merge(channels)


def color_correction(
    image: np.ndarray,
    method: Literal["gray_world", "white_balance", "histogram"] = "gray_world",
    **kwargs,
) -> np.ndarray:
    """
    Apply color correction to an underwater image.

    Args:
        image: BGR image (uint8).
        method: "gray_world" | "white_balance" | "histogram".
        **kwargs: Passed to percentile method for "white_balance"
                  (e.g. low_percentile=2.0, high_percentile=98.0).

    Returns:
        Color-corrected BGR image (uint8).
    """
    if method == "gray_world":
        return color_correction_gray_world(image)
    if method == "white_balance":
        return color_correction_white_balance_percentile(image, **kwargs)
    if method == "histogram":
        return color_correction_histogram_equalization(image)
    raise ValueError(f"Unknown color_correction method: {method}")


# -----------------------------------------------------------------------------
# Contrast enhancement
# -----------------------------------------------------------------------------


def contrast_enhancement_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_lab: bool = True,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (OpenCV CLAHE).

    When use_lab=True, CLAHE is applied only on the L (luminance) channel
    in LAB space to boost contrast without shifting colors—recommended for
    underwater images.

    Args:
        image: BGR image (uint8).
        clip_limit: Contrast limit (e.g. 1.5–4.0). Higher = stronger contrast.
        tile_grid_size: Grid size for local histograms (e.g. (8, 8)).
        use_lab: If True, apply on L in LAB; else on grayscale.

    Returns:
        Contrast-enhanced BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if use_lab:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def contrast_enhancement_histogram(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization on luminance (LAB L channel).

    Improves overall contrast with a single global transform. For more local
    control and less noise amplification, use contrast_enhancement_clahe.

    Args:
        image: BGR image (uint8).

    Returns:
        Contrast-enhanced BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def contrast_enhancement(
    image: np.ndarray,
    method: Literal["clahe", "histogram"] = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_lab: bool = True,
) -> np.ndarray:
    """
    Apply contrast enhancement to an underwater image.

    Args:
        image: BGR image (uint8).
        method: "clahe" (local, recommended) or "histogram" (global).
        clip_limit: For CLAHE only.
        tile_grid_size: For CLAHE only.
        use_lab: For CLAHE only; apply on L channel when True.

    Returns:
        Contrast-enhanced BGR image (uint8).
    """
    if method == "clahe":
        return contrast_enhancement_clahe(
            image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            use_lab=use_lab,
        )
    if method == "histogram":
        return contrast_enhancement_histogram(image)
    raise ValueError(f"Unknown contrast_enhancement method: {method}")


# -----------------------------------------------------------------------------
# Combined pipeline
# -----------------------------------------------------------------------------


def enhance_underwater(
    image: np.ndarray,
    color_method: Literal["gray_world", "white_balance", "histogram", "none"] = "gray_world",
    contrast_method: Literal["clahe", "histogram", "none"] = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    **color_kwargs,
) -> np.ndarray:
    """
    Full underwater enhancement: color correction followed by contrast enhancement.

    Order: color correction first to restore channel balance, then contrast
    enhancement on the corrected image.

    Args:
        image: BGR image (uint8).
        color_method: Color correction method or "none" to skip.
        contrast_method: Contrast method or "none" to skip.
        clip_limit: CLAHE clip limit (used if contrast_method=="clahe").
        tile_grid_size: CLAHE grid size (used if contrast_method=="clahe").
        **color_kwargs: Passed to color_correction (e.g. for "white_balance").

    Returns:
        Enhanced BGR image (uint8).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    out = image.copy()

    if color_method != "none":
        out = color_correction(out, method=color_method, **color_kwargs)

    if contrast_method != "none":
        out = contrast_enhancement(
            out,
            method=contrast_method,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    return out
