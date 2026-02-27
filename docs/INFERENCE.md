# Image and Video Inference – How to Run

Use the trained YOLOv8 model to detect marine waste in new images or videos.

**Requirements:** Trained weights (e.g. `runs/detect/train/weights/best.pt`) and `ultralytics` installed (`pip install ultralytics`).

---

## Image inference

**Script:** `inference/detect_image.py`  
**Usage:** Run from the **project root** (the folder that contains `inference/`, `dataset/`, etc.).

### Single image

```bash
python -m inference.detect_image --source path/to/image.jpg --weights runs/detect/train/weights/best.pt --output results
```

- **`--source` / `-s`** – Path to one image (e.g. `.jpg`, `.png`).
- **`--weights` / `-w`** – Path to the model weights file (e.g. `best.pt` or `last.pt`).
- **`--output` / `-o`** – Directory where the annotated image is saved (default: `results`).  
  Output is written under `results/image/` (e.g. `results/image/image.jpg`).

### Folder of images

Process every image in a folder:

```bash
python -m inference.detect_image --source dataset/raw --weights runs/detect/train/weights/best.pt --output results
```

All images in `dataset/raw` are run through the model; annotated images are saved under `--output`.

### Optional arguments

| Argument        | Default | Description |
|----------------|--------|-------------|
| `--conf`       | 0.5    | Confidence threshold (0–1). Detections below this are hidden. |
| `--iou`        | 0.45   | IoU threshold for NMS (overlapping boxes). |
| `--preprocess` | off    | Apply underwater preprocessing (color correction, CLAHE, dehazing) before detection. |
| `--no-save`    | off    | Do not save images (only run the model). |

**Example with preprocessing and higher confidence:**

```bash
python -m inference.detect_image -s underwater_photo.jpg -w runs/detect/train/weights/best.pt -o results --preprocess --conf 0.6
```

---

## Video inference

**Script:** `inference/detect_video.py`  
**Usage:** Run from the **project root**.

### Video file

```bash
python -m inference.detect_video --source path/to/video.mp4 --weights runs/detect/train/weights/best.pt --output results
```

- **`--source` / `-s`** – Path to the video file (e.g. `.mp4`, `.avi`) or `0` for the default webcam.
- **`--weights` / `-w`** – Path to the model weights (e.g. `best.pt`).
- **`--output` / `-o`** – Directory for the annotated video (default: `results`).  
  The saved video is usually under `results/video/` (e.g. `results/video/video.mp4` or similar).

### Webcam

```bash
python -m inference.detect_video --source 0 --weights runs/detect/train/weights/best.pt --output results
```

Use `--source 1` (or `2`, …) for another camera.

### Optional arguments

| Argument        | Default | Description |
|----------------|--------|-------------|
| `--conf`       | 0.5    | Confidence threshold (0–1). |
| `--iou`        | 0.45   | IoU threshold for NMS. |
| `--preprocess` | off    | Preprocess each frame (color correction, CLAHE, dehazing). Slower but can help underwater footage. |
| `--no-save`    | off    | Do not write the output video. |

**Example with preprocessing:**

```bash
python -m inference.detect_video -s dive.mp4 -w runs/detect/train/weights/best.pt -o results --preprocess --conf 0.5
```

---

## Defaults from config

If `config/config.yaml` exists and has an `inference` section, the scripts use:

- `confidence_threshold` → default `--conf`
- `iou_threshold` → default `--iou`

You can still override them with `--conf` and `--iou` on the command line.

---

## Quick reference

| Task              | Command |
|-------------------|--------|
| One image         | `python -m inference.detect_image -s image.jpg -w runs/detect/train/weights/best.pt` |
| Folder of images  | `python -m inference.detect_image -s dataset/raw -w runs/detect/train/weights/best.pt` |
| Video file        | `python -m inference.detect_video -s video.mp4 -w runs/detect/train/weights/best.pt` |
| Webcam            | `python -m inference.detect_video -s 0 -w runs/detect/train/weights/best.pt` |
| With preprocessing | Add `--preprocess` to either command. |

Output is written under the directory given by `--output` (default: `results/`).
