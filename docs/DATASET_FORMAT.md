# YOLO-Style Dataset and dataset.yaml Format

This project uses the same dataset format as **Ultralytics YOLOv8**: a `data.yaml` (or `dataset.yaml`) file plus images and label files in YOLO format.

---

## 1. The dataset.yaml (data.yaml) format

The YAML file tells the trainer where to find images and labels and how many classes there are.

### Required and optional fields

| Field   | Required | Description |
|--------|----------|-------------|
| **path** | Yes | Root directory of the dataset. Can be absolute or relative to the YAML file. If the YAML lives in the dataset folder (e.g. `dataset/data.yaml`), use `path: .` so the root is that folder. |
| **train** | Yes | Where to find **training** images. See “Split paths” below. |
| **val**   | Yes | Where to find **validation** images. |
| **test**  | No  | Where to find **test** images (optional). |
| **nc**    | Yes | Number of classes (integer). Must match the length of `names`. |
| **names** | Yes | List of class names. The **index** of each name is the **class_id** used in the `.txt` label files (0-based). |

### Split paths (train / val / test)

Each of `train`, `val`, and `test` can be:

1. **A directory path** (relative to `path` or absolute)  
   All image files in that directory are used for that split.  
   Example: `train: images/train`

2. **A .txt file** (relative to `path` or absolute)  
   Each line is one image path (absolute or relative to `path`).  
   Example: `train: splits/train.txt`

### Example data.yaml

```yaml
# Dataset root (use "." if this file is inside the dataset folder)
path: .

# Where images are for each split (directories or .txt file lists)
train: images/train
val:   images/val
test:  images/test

# Number of classes
nc: 5

# Class names; index = class_id in annotation files
names:
  - plastic_bottle
  - plastic_bag
  - fishing_net
  - can
  - other_waste
```

### Alternative: names as a dictionary

Ultralytics (and this project’s loader) also accept class names as a dict:

```yaml
names:
  0: plastic_bottle
  1: plastic_bag
  2: fishing_net
  3: can
  4: other_waste
```

---

## 2. Directory layout

Two common layouts are supported.

### Layout A: images/ and labels/ folders

Images and labels are in parallel directories with the same structure and filenames (only extension changes: image → `.jpg`, label → `.txt`).

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── val/
│   │   └── img003.jpg
│   └── test/
│       └── img004.jpg
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── img002.txt
    ├── val/
    │   └── img003.txt
    └── test/
        └── img004.txt
```

With this layout, `data.yaml` would look like:

```yaml
path: .
train: images/train
val:   images/val
test:  images/test
nc: 5
names: [plastic_bottle, plastic_bag, fishing_net, can, other_waste]
```

The loader will look for labels at `labels/train/img001.txt`, etc., when the image is at `images/train/img001.jpg`.

### Layout B: split files (train.txt, val.txt, test.txt)

Images and labels can live anywhere; text files list the image paths.

```
dataset/
├── data.yaml
├── splits/
│   ├── train.txt   # one image path per line
│   ├── val.txt
│   └── test.txt
├── processed/      # or any folder
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── annotations/
    ├── img001.txt
    ├── img002.txt
    └── ...
```

Example `splits/train.txt` (paths relative to dataset root or absolute):

```
processed/img001.jpg
processed/img002.jpg
```

For this layout, labels are assumed to be in the same directory as the image with the same base name and `.txt` extension (e.g. `processed/img001.jpg` → `processed/img001.txt`), or in a parallel `labels/` structure if the path contains an `images/` segment.

---

## 3. YOLO annotation format (.txt files)

- **One `.txt` file per image**, same base name as the image (e.g. `img001.jpg` → `img001.txt`).
- **One line per object**.
- **Line format:**  
  `class_id x_center y_center width height`  
  - All coordinates are **normalized** in the range **0–1** (relative to image width and height).
  - `x_center`, `y_center`: center of the box.  
  - `width`, `height`: width and height of the box.  
  - `class_id`: integer index into the `names` list in `data.yaml` (0-based).

Example `img001.txt` (two objects: class 0 and class 2):

```
0 0.45 0.32 0.20 0.40
2 0.70 0.55 0.15 0.25
```

No object in the image → empty `.txt` file or no lines (only blank lines/comments).

---

## 4. Loading the dataset in code

Use the training loader:

```python
from training.yolo_dataset import (
    load_dataset_yaml,
    get_split_image_paths,
    load_yolo_annotations,
    YOLODataset,
    validate_dataset_yaml,
)

# Load config
data = load_dataset_yaml("dataset/data.yaml")

# Get image paths for a split
train_paths = get_split_image_paths(data, "train")

# Load annotations for one image (you need to resolve label path; YOLODataset does this)
# Using the helper class:
ds = YOLODataset.from_yaml_path("dataset/data.yaml", split="train")
for i in range(len(ds)):
    image_path, boxes = ds[i]  # boxes: list of (class_id, x_center, y_center, w, h)
```

Validate before training:

```python
ok, errors = validate_dataset_yaml("dataset/data.yaml")
if not ok:
    print("Errors:", errors)
```

An example `data.yaml` with comments is in **config/data.yaml.example**. Copy it to `dataset/data.yaml` and adjust paths and class names to match your marine waste dataset.
