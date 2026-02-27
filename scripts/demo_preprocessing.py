"""
Demo script for preprocessing pipeline.

Creates a dummy underwater-like image and applies preprocessing.
Run from project root: python scripts/demo_preprocessing.py

Useful for testing preprocessing without real dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from preprocessing import preprocess_image, apply_gray_world, apply_clahe, apply_dark_channel_dehaze


def create_dummy_underwater_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic underwater-like image for testing.
    
    Simulates blue/green dominance and low contrast.
    """
    # Start with a simple gradient + noise
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 180 + np.random.randint(-20, 20, (height, width))  # Blue - dominant
    img[:, :, 1] = 160 + np.random.randint(-20, 20, (height, width))  # Green
    img[:, :, 2] = 80 + np.random.randint(-15, 15, (height, width))   # Red - absorbed
    
    # Add a "waste" object (white rectangle) to simulate plastic
    cv2.rectangle(img, (200, 150), (400, 300), (220, 220, 220), -1)
    cv2.rectangle(img, (250, 200), (350, 250), (200, 200, 200), -1)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("Creating dummy underwater image...")
    dummy = create_dummy_underwater_image()
    
    output_dir = Path("results/demo_preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    cv2.imwrite(str(output_dir / "00_original.png"), dummy)
    print(f"Saved: {output_dir / '00_original.png'}")
    
    # Step-by-step
    step1 = apply_gray_world(dummy)
    cv2.imwrite(str(output_dir / "01_color_corrected.png"), step1)
    print(f"Saved: {output_dir / '01_color_corrected.png'}")
    
    step2 = apply_clahe(step1, clip_limit=2.0, tile_grid_size=(8, 8))
    cv2.imwrite(str(output_dir / "02_clahe.png"), step2)
    print(f"Saved: {output_dir / '02_clahe.png'}")
    
    step3 = apply_dark_channel_dehaze(step2, omega=0.95, t0=0.1)
    cv2.imwrite(str(output_dir / "03_dehazed.png"), step3)
    print(f"Saved: {output_dir / '03_dehazed.png'}")
    
    # Full pipeline
    full = preprocess_image(dummy, color_correct=True, clahe=True, dehaze=True)
    cv2.imwrite(str(output_dir / "04_full_pipeline.png"), full)
    print(f"Saved: {output_dir / '04_full_pipeline.png'}")
    
    print("\nDone! Check results/demo_preprocessing/ for output images.")


if __name__ == "__main__":
    main()
