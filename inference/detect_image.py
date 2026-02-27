"""
Single image (or directory of images) inference for marine waste detection.

Loads a trained YOLOv8 model and runs detection. Optionally applies underwater
preprocessing (color correction, CLAHE, dehazing) before inference.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")


def _load_and_preprocess_if_needed(image_path: str, preprocess: bool):
    """Load image and optionally run underwater preprocessing."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if preprocess:
        from preprocessing.pipeline import preprocess_image
        img = preprocess_image(img, color_correct=True, clahe=True, dehaze=True)
    return img


def detect_image(
    source: str,
    weights: str,
    output_dir: str = "results",
    conf: float = 0.5,
    iou: float = 0.45,
    preprocess: bool = False,
    save: bool = True,
    show_labels: bool = True,
) -> list:
    """
    Run marine waste detection on an image or a directory of images.

    Args:
        source: Path to a single image or to a folder of images (.jpg, .png, etc.).
        weights: Path to model weights (e.g. runs/detect/train/weights/best.pt).
        output_dir: Directory where annotated images are saved (when save=True).
        conf: Confidence threshold (0–1). Detections below this are discarded.
        iou: IoU threshold for NMS (overlapping boxes).
        preprocess: If True, apply underwater enhancement before inference.
        save: If True, save annotated images to output_dir.
        show_labels: If True, draw class labels and confidence on boxes.

    Returns:
        List of Ultralytics Results objects (one per image).
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    YOLO = _get_yolo()
    model = YOLO(str(weights_path))

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Single image
    if source_path.is_file():
        if preprocess:
            img = _load_and_preprocess_if_needed(str(source_path), preprocess=True)
            results = model.predict(
                source=img,
                conf=conf,
                iou=iou,
                save=save,
                project=output_dir,
                name="image",
                exist_ok=True,
                show_labels=show_labels,
            )
        else:
            results = model.predict(
                source=str(source_path),
                conf=conf,
                iou=iou,
                save=save,
                project=output_dir,
                name="image",
                exist_ok=True,
                show_labels=show_labels,
            )
        return list(results)

    # Directory of images
    if source_path.is_dir():
        from utils.helpers import get_image_paths
        image_paths = get_image_paths(str(source_path), recursive=False)
        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {source}")
        all_results = []
        for img_path in image_paths:
            if preprocess:
                img = _load_and_preprocess_if_needed(str(img_path), preprocess=True)
                res = model.predict(
                    source=img,
                    conf=conf,
                    iou=iou,
                    save=save,
                    project=output_dir,
                    name="image",
                    exist_ok=True,
                    show_labels=show_labels,
                )
            else:
                res = model.predict(
                    source=str(img_path),
                    conf=conf,
                    iou=iou,
                    save=save,
                    project=output_dir,
                    name="image",
                    exist_ok=True,
                    show_labels=show_labels,
                )
            all_results.extend(list(res))
        return all_results

    raise ValueError(f"Source must be a file or directory: {source}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect marine waste in an image or folder of images (YOLOv8)",
    )
    parser.add_argument("--source", "-s", required=True, help="Path to image or folder of images")
    parser.add_argument("--weights", "-w", required=True, help="Path to model weights (e.g. best.pt)")
    parser.add_argument("--output", "-o", default="results", help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0–1)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--preprocess", action="store_true", help="Apply underwater preprocessing before detection")
    parser.add_argument("--no-save", action="store_true", help="Do not save output images")
    args = parser.parse_args()

    # Optional config defaults
    try:
        from utils.helpers import load_config
        cfg = load_config()
        inf = cfg.get("inference", {})
        conf = getattr(args, "conf", None) or inf.get("confidence_threshold", 0.5)
        iou = getattr(args, "iou", None) or inf.get("iou_threshold", 0.45)
    except Exception:
        conf, iou = args.conf, args.iou

    results = detect_image(
        source=args.source,
        weights=args.weights,
        output_dir=args.output,
        conf=conf,
        iou=iou,
        preprocess=args.preprocess,
        save=not args.no_save,
    )
    print(f"Processed {len(results)} image(s). Output: {args.output}")


if __name__ == "__main__":
    main()
