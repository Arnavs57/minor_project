# Underwater Image Processing for Marine Waste and Pollution Detection Using Deep Learning

**College Major Project** | Computer Vision & Deep Learning

## Project Overview

This project aims to **detect and localize marine waste** (plastic bottles, bags, fishing nets, cans, etc.) from underwater images and videos using deep learning. It addresses key challenges in underwater imaging:

- **Low visibility and blur** – Light scattering reduces clarity
- **Blue/green color dominance** – Wavelength-dependent absorption distorts colors
- **Noise and light absorption** – Degraded image quality affects detection

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Image Preprocessing | OpenCV |
| Deep Learning | PyTorch |
| Object Detection | YOLOv8 (Ultralytics) |
| Training Environment | Google Colab / Local GPU |

## Project Structure

```
underwater-marine-waste-detection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   ├── config.yaml           # Hyperparameters and paths
│   └── data.yaml.example     # Example dataset.yaml (see docs/DATASET_FORMAT.md)
├── docs/
│   ├── DATASET_FORMAT.md     # dataset.yaml format and YOLO annotation layout
│   ├── INFERENCE.md          # How to run image and video inference
│   └── METRICS.md            # mAP, IoU, precision, recall explained
├── dataset/
│   ├── raw/                  # Original underwater images
│   ├── processed/            # Preprocessed (enhanced) images
│   ├── annotations/          # YOLO format labels
│   └── splits/               # train/val/test splits
├── preprocessing/            # Image enhancement module
│   ├── __init__.py
│   ├── color_correction.py   # Color balance correction
│   ├── contrast_enhancement.py  # CLAHE
│   ├── dehazing.py           # Dark channel prior dehazing
│   └── pipeline.py           # Full preprocessing pipeline
├── training/                 # Model training
│   ├── __init__.py
│   ├── train.py              # YOLOv8 training script
│   ├── prepare_dataset.py    # Dataset preparation
│   └── yolo_dataset.py       # YOLO dataset.yaml loader
├── inference/                # Detection on new data
│   ├── __init__.py
│   ├── detect_image.py       # Single image inference
│   └── detect_video.py       # Video inference
├── evaluation/               # Performance metrics
│   ├── __init__.py
│   └── metrics.py            # mAP, precision, recall
├── scripts/
│   └── demo_preprocessing.py # Demo preprocessing on dummy image
└── utils/
    ├── __init__.py
    └── helpers.py            # Common utilities
```

## Setup

### 1. Clone and Install

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Preprocessing (Optional)

Run the demo to verify preprocessing works (creates dummy underwater image):

```bash
python scripts/demo_preprocessing.py
```

Output images are saved in `results/demo_preprocessing/`.

### 3. Dataset

Place your underwater marine waste images and YOLO-format annotations as described in **docs/DATASET_FORMAT.md**. Use a **dataset.yaml** (or **data.yaml**) to point to train/val/test splits and class names; an example is in **config/data.yaml.example**.

- **dataset.yaml format:** `path`, `train`, `val`, `test`, `nc`, `names`. Splits can be directories or `.txt` files listing image paths.
- **YOLO labels:** One `.txt` per image, each line: `class_id x_center y_center width height` (normalized 0–1).

Load the dataset in code with `training.yolo_dataset` (e.g. `YOLODataset.from_yaml_path("dataset/data.yaml", split="train")`).

**Public datasets to consider:**
- [TrashCan](https://github.com/WasabiPhong/TrashCan) – Underwater trash dataset
- [Marine Debris Dataset](https://www.kaggle.com/datasets) – Search for marine/plastic waste
- [OpenLitterMap](https://openlittermap.com/) – Crowdsourced litter data

### 4. Preprocessing

```bash
python -m preprocessing.pipeline --input dataset/raw --output dataset/processed
```

### 5. Training

```bash
python -m training.train --data dataset/data.yaml --epochs 100
```

### 6. Inference

After training, use the saved weights (e.g. `runs/detect/train/weights/best.pt`) to run detection on new images or videos. **See [docs/INFERENCE.md](docs/INFERENCE.md) for full details.**

**Image** (single file or folder):

```bash
python -m inference.detect_image --source path/to/image.jpg --weights runs/detect/train/weights/best.pt --output results
python -m inference.detect_image --source dataset/raw --weights runs/detect/train/weights/best.pt --output results
```

**Video** (file or webcam `0`):

```bash
python -m inference.detect_video --source path/to/video.mp4 --weights runs/detect/train/weights/best.pt --output results
python -m inference.detect_video --source 0 --weights runs/detect/train/weights/best.pt --output results
```

Add `--preprocess` to apply underwater enhancement before detection. Use `--conf` and `--iou` to tune thresholds.

## Preprocessing Techniques

| Technique | Purpose |
|-----------|---------|
| **Color Correction** | Restore natural colors (Gray World, Retinex) |
| **CLAHE** | Improve local contrast in low-visibility regions |
| **Dehazing** | Reduce scattering/haze using Dark Channel Prior |

## Evaluation Metrics

Implemented in **`evaluation/metrics.py`** (see **docs/METRICS.md** for full explanations):

- **IoU** (Intersection over Union) – Overlap between predicted and ground-truth boxes; used as the match criterion (e.g. IoU ≥ 0.5 = True Positive).
- **Precision** – TP / (TP + FP): proportion of detections that are correct.
- **Recall** – TP / (TP + FN): proportion of ground-truth objects detected.
- **mAP@0.5** – Mean Average Precision at IoU threshold 0.5 (main detection score).
- **mAP@0.5:0.95** – COCO-style: average of mAP at IoU 0.5, 0.55, …, 0.95.

Run a quick demo: `python -m evaluation.metrics` (saves a metrics plot to `evaluation_results/`).

## License

MIT License – Free for academic and educational use.

## References

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Underwater Image Enhancement: Dark Channel Prior, CLAHE, Color Correction
