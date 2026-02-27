"""
YOLOv8 training script for marine waste detection.

Uses transfer learning from pretrained YOLOv8 weights (PyTorch backend via Ultralytics).
Supports local GPU or Google Colab. Saves best.pt and last.pt to project/name/weights/.
"""

import argparse
import sys
from pathlib import Path

# Project root for config/utils
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def _get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics is required for YOLOv8 training. Install with: pip install ultralytics"
        )


def load_training_config(config_path: str = "config/config.yaml"):
    """Load training section from project config.yaml if it exists."""
    try:
        import yaml
        p = _PROJECT_ROOT / config_path
        if p.exists():
            with open(p, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("training", {})
    except Exception:
        pass
    return {}


def train(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    batch: int = 16,
    img_size: int = 640,
    project: str = "runs/detect",
    name: str = "train",
    patience: int = 20,
    device: str = "",
    exist_ok: bool = False,
    resume: bool = False,
    **kwargs,
):
    """
    Train YOLOv8 model for marine waste detection using pretrained weights.

    Pretrained weights (e.g. yolov8n.pt) are downloaded automatically on first use.
    Training uses PyTorch; set device to "0" for GPU or "cpu" for CPU.

    Args:
        data_yaml: Path to dataset data.yaml (train/val paths, nc, names).
        model: Pretrained model path or name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.).
        epochs: Number of training epochs.
        batch: Batch size (-1 for auto from model).
        img_size: Input image size (square).
        project: Project directory for saving runs.
        name: Run name (saves to project/name/).
        patience: Early stopping patience (epochs without mAP improvement).
        device: Device string ("" = auto, "0" = GPU 0, "cpu" = CPU).
        exist_ok: Overwrite existing project/name if True.
        resume: Resume training from last.pt in project/name if True.
        **kwargs: Additional arguments passed to model.train() (e.g. optimizer, lr0, amp).
    """
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = _PROJECT_ROOT / data_yaml
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. "
            "Create dataset/data.yaml (see config/data.yaml.example and docs/DATASET_FORMAT.md)."
        )

    # Optional: validate dataset layout before training
    try:
        from training.yolo_dataset import validate_dataset_yaml
        ok, errors = validate_dataset_yaml(data_path)
        if not ok and errors:
            print("Dataset validation warnings:", errors)
    except Exception:
        pass

    # Load pretrained YOLOv8 model (downloads if needed)
    YOLO = _get_yolo()
    yolo_model = YOLO(model)

    train_args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": img_size,
        "project": project,
        "name": name,
        "patience": patience,
        "exist_ok": exist_ok,
        "resume": resume,
    }
    if device:
        train_args["device"] = device
    train_args.update(kwargs)

    results = yolo_model.train(**train_args)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for marine waste detection (PyTorch, pretrained weights)"
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to data.yaml (dataset config with train/val, nc, names)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained model: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--img-size", type=int, default=None, help="Input image size")
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Project directory for saving runs",
    )
    parser.add_argument("--name", default="train", help="Run name (output: project/name/)")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--device",
        default="",
        help='Device: "" (auto), "0" (GPU 0), "cpu"',
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing project/name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt",
    )
    args = parser.parse_args()

    # Merge with config.yaml if present
    cfg = load_training_config()
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    batch = args.batch if args.batch is not None else cfg.get("batch_size", 16)
    img_size = args.img_size if args.img_size is not None else cfg.get("img_size", 640)
    patience = args.patience if args.patience is not None else cfg.get("patience", 20)
    model = args.model or cfg.get("model", "yolov8n.pt")

    train(
        data_yaml=args.data,
        model=model,
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        project=args.project,
        name=args.name,
        patience=patience,
        device=args.device,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
