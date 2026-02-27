"""
Dataset preparation for YOLOv8 training.

Prepares train/val/test splits and ensures YOLO format compatibility.
YOLO format: one .txt per image, each line: class_id center_x center_y width height (normalized 0-1)
"""

import argparse
import random
from pathlib import Path

# TODO: Implement dataset preparation
# 1. Load images from dataset/processed (or raw)
# 2. Load corresponding annotations from dataset/annotations
# 3. Create train/val/test splits (e.g., 70/20/10)
# 4. Write split files to dataset/splits/ (train.txt, val.txt, test.txt)
# 5. Create data.yaml for Ultralytics YOLOv8 with:
#    - path, train, val, test paths
#    - nc (number of classes)
#    - names (class names)
# 6. Validate annotation format (class_id x_center y_center w h)

# Placeholder: Example data.yaml structure for YOLOv8
# path: ../dataset
# train: splits/train.txt
# val: splits/val.txt
# test: splits/test.txt
# nc: 5
# names: ['plastic_bottle', 'plastic_bag', 'fishing_net', 'can', 'other_waste']


def create_splits(
    images_dir: str,
    annotations_dir: str,
    splits_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Create train/val/test splits and write to splits directory.
    
    TODO: Implement this function.
    - Get all image paths (with matching annotation files)
    - Shuffle with seed
    - Split according to ratios
    - Write train.txt, val.txt, test.txt with image paths (relative to project root)
    """
    raise NotImplementedError("TODO: Implement create_splits")


def create_data_yaml(
    output_path: str,
    dataset_path: str,
    class_names: list,
) -> None:
    """
    Create data.yaml for Ultralytics YOLOv8.
    
    TODO: Implement this function.
    """
    raise NotImplementedError("TODO: Implement create_data_yaml")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training")
    parser.add_argument("--images", default="dataset/processed", help="Images directory")
    parser.add_argument("--annotations", default="dataset/annotations", help="Annotations directory")
    parser.add_argument("--splits", default="dataset/splits", help="Output splits directory")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # TODO: Call create_splits and create_data_yaml
    print("TODO: Implement dataset preparation. See prepare_dataset.py")


if __name__ == "__main__":
    main()
