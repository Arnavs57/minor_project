"""
YOLO-style dataset loading and data.yaml support.

Loads dataset configuration from a dataset.yaml file (Ultralytics-style)
and provides access to image paths, annotation paths, and parsed bounding boxes.
See docs/DATASET_FORMAT.md for the dataset.yaml format and directory layout.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

# Default image extensions for YOLO datasets
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# data.yaml loading and validation
# -----------------------------------------------------------------------------


def load_dataset_yaml(yaml_path: Union[str, Path]) -> Dict:
    """
    Load and resolve a YOLO dataset.yaml file.

    Paths in the YAML can be relative to the YAML file's directory or absolute.
    Resolves 'path', and entries like 'train', 'val', 'test' relative to 'path'
    when they are relative.

    Args:
        yaml_path: Path to data.yaml (e.g. dataset/data.yaml).

    Returns:
        Dictionary with keys: path, train, val, test (optional), nc, names.
        Paths are resolved as absolute Path objects where applicable.
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("Dataset YAML is empty")

    base = yaml_path.parent
    # Dataset root: 'path' is relative to yaml dir or absolute
    raw_path = data.get("path", ".")
    path = (base / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    data["path"] = path

    # Resolve train/val/test relative to dataset root (path)
    for split in ("train", "val", "test"):
        if split not in data or data[split] is None:
            continue
        raw = data[split]
        if isinstance(raw, str):
            # Can be a directory path or a .txt file listing image paths
            p = path / raw if not Path(raw).is_absolute() else Path(raw)
            data[split] = p.resolve()
        elif isinstance(raw, list):
            data[split] = [path / r if not Path(r).is_absolute() else Path(r) for r in raw]
            data[split] = [p.resolve() for p in data[split]]

    if "nc" not in data and "names" in data:
        data["nc"] = len(data["names"])
    if "names" in data and isinstance(data["names"], dict):
        # Ultralytics allows names as {0: 'class0', 1: 'class1'}
        data["names"] = [data["names"][i] for i in sorted(data["names"])]

    return data


def get_split_image_paths(
    data: Dict,
    split: str,
    extensions: Optional[set] = None,
) -> List[Path]:
    """
    Get list of image paths for a split (train, val, or test).

    The split entry in data can be:
    - A path to a directory: all images inside (non-recursive) are used.
    - A path to a .txt file: each line is an image path (absolute or relative to dataset path).

    Args:
        data: Result of load_dataset_yaml().
        split: One of "train", "val", "test".
        extensions: Set of allowed image extensions; default IMAGE_EXTENSIONS.

    Returns:
        Sorted list of absolute Path objects to images.
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    if split not in data or data[split] is None:
        return []

    entry = data[split]
    base = data["path"]
    paths: List[Path] = []

    if isinstance(entry, Path):
        entry = entry
    else:
        entry = Path(entry) if not isinstance(entry, list) else entry

    if isinstance(entry, list):
        for p in entry:
            p = Path(p)
            if not p.is_absolute():
                p = base / p
            if p.suffix.lower() in extensions:
                paths.append(p.resolve())
    elif entry.is_file():
        if entry.suffix.lower() == ".txt":
            with open(entry, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = Path(line)
                    if not p.is_absolute():
                        p = base / line
                    if p.suffix.lower() in extensions:
                        paths.append(p.resolve())
        else:
            if entry.suffix.lower() in extensions:
                paths.append(entry.resolve())
    elif entry.is_dir():
        for p in sorted(entry.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                paths.append(p.resolve())

    return sorted(set(paths))


def image_path_to_label_path(
    image_path: Path,
    images_dir: Optional[Path] = None,
    labels_dir: Optional[Path] = None,
) -> Path:
    """
    Get the annotation file path for an image (YOLO convention).

    By convention, labels live in a parallel directory with the same structure
    and the same filename with .txt extension. Example:
      images/train/photo1.jpg -> labels/train/photo1.txt

    If images_dir and labels_dir are given, image_path is assumed relative to
    images_dir and the label path is built under labels_dir. Otherwise,
    the label path is the same directory as the image with .txt extension.

    Args:
        image_path: Path to the image file.
        images_dir: Optional root of images (e.g. path / "images").
        labels_dir: Optional root of labels (e.g. path / "labels").

    Returns:
        Path to the .txt annotation file.
    """
    image_path = Path(image_path)
    if images_dir is not None and labels_dir is not None:
        images_dir = Path(images_dir).resolve()
        labels_dir = Path(labels_dir).resolve()
        try:
            rel = image_path.resolve().relative_to(images_dir)
        except ValueError:
            rel = image_path.name
        return (labels_dir / rel).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_annotation_line(line: str) -> Tuple[int, float, float, float, float]:
    """
    Parse a single line from a YOLO annotation file.

    Format: "class_id x_center y_center width height" (space-separated, normalized 0-1).

    Args:
        line: One line of the .txt file.

    Returns:
        (class_id, x_center, y_center, width, height).
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO line must have 5 values, got: {line!r}")
    class_id = int(parts[0])
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return class_id, x, y, w, h


def load_yolo_annotations(
    label_path: Union[str, Path],
    return_xyxy: bool = False,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Load all bounding boxes from a YOLO format .txt file.

    Args:
        label_path: Path to the annotation .txt file.
        return_xyxy: If True, convert (x_center, y_center, w, h) to (x1, y1, x2, y2)
                    in normalized coordinates (0-1). Default False keeps center format.

    Returns:
        List of (class_id, x_center, y_center, width, height) or
        (class_id, x1, y1, x2, y2) if return_xyxy=True. Empty list if file missing.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                class_id, xc, yc, w, h = parse_yolo_annotation_line(line)
            except (ValueError, TypeError):
                continue
            if return_xyxy:
                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2
                boxes.append((class_id, x1, y1, x2, y2))
            else:
                boxes.append((class_id, xc, yc, w, h))
    return boxes


# -----------------------------------------------------------------------------
# High-level dataset interface
# -----------------------------------------------------------------------------


class YOLODataset:
    """
    YOLO-style dataset backed by a dataset.yaml file.

    Use load_dataset_yaml() to get the config dict, then pass it here,
    or use from_yaml_path() to load from file.
    """

    def __init__(
        self,
        data: Dict,
        split: str = "train",
        images_dir_key: Optional[str] = None,
        labels_dir_key: Optional[str] = None,
    ):
        """
        Args:
            data: Config dict from load_dataset_yaml() (must include "path", split key).
            split: "train", "val", or "test".
            images_dir_key: Optional key in data for images root (e.g. "train").
            labels_dir_key: Optional key in data for labels root (e.g. "train_labels").
                           If not set, label path is inferred by replacing
                           "images" with "labels" in the image path, or same dir as image.
        """
        self.data = data
        self.split = split
        self.images_dir_key = images_dir_key
        self.labels_dir_key = labels_dir_key
        self._image_paths: Optional[List[Path]] = None
        self._base = data["path"]

    @classmethod
    def from_yaml_path(
        cls,
        yaml_path: Union[str, Path],
        split: str = "train",
        **kwargs,
    ) -> "YOLODataset":
        """Build dataset from a dataset.yaml file path."""
        data = load_dataset_yaml(yaml_path)
        return cls(data, split=split, **kwargs)

    @property
    def image_paths(self) -> List[Path]:
        """List of image paths for this split."""
        if self._image_paths is None:
            self._image_paths = get_split_image_paths(self.data, self.split)
        return self._image_paths

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class names (index = class_id)."""
        return self.data.get("names", [])

    @property
    def num_classes(self) -> int:
        """Number of classes (nc)."""
        return self.data.get("nc", 0)

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_label_path(self, image_path: Path) -> Path:
        """Get the .txt label path for an image (same stem, labels dir or same dir)."""
        # Common layouts:
        # 1) path/images/train/img.jpg -> path/labels/train/img.txt
        # 2) path/train/img.jpg -> path/train/img.txt (labels next to images)
        img_path = Path(image_path).resolve()
        base = Path(self._base).resolve()
        try:
            rel = img_path.relative_to(base)
        except ValueError:
            return img_path.with_suffix(".txt")
        parts = rel.parts
        if len(parts) >= 2 and parts[0].lower() == "images":
            # path/images/train/img.jpg -> path/labels/train/img.txt
            new_rel = Path("labels") / Path(*parts[1:]).with_suffix(".txt")
            return base / new_rel
        # Same directory as image
        return img_path.with_suffix(".txt")

    def load_annotations(self, image_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """Load YOLO annotations for one image (class_id, x_center, y_center, w, h) normalized."""
        label_path = self.get_label_path(image_path)
        return load_yolo_annotations(label_path, return_xyxy=False)

    def __getitem__(self, index: int) -> Tuple[Path, List[Tuple[int, float, float, float, float]]]:
        """Return (image_path, list of (class_id, xc, yc, w, h)) for the index-th sample."""
        img_path = self.image_paths[index]
        boxes = self.load_annotations(img_path)
        return img_path, boxes


def validate_dataset_yaml(yaml_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Check that the dataset.yaml and referenced paths/files exist.

    Returns:
        (all_ok, list of error/warning messages).
    """
    errors: List[str] = []
    try:
        data = load_dataset_yaml(yaml_path)
    except Exception as e:
        return False, [str(e)]

    base = data["path"]
    if not base.exists():
        errors.append(f"Dataset path does not exist: {base}")

    for split in ("train", "val", "test"):
        if split not in data or data[split] is None:
            continue
        entry = data[split]
        if isinstance(entry, list):
            for p in entry:
                if not Path(p).exists():
                    errors.append(f"Split {split} path missing: {p}")
        else:
            p = Path(entry)
            if not p.exists():
                errors.append(f"Split {split} path missing: {p}")

    if data.get("nc", 0) <= 0:
        errors.append("'nc' (number of classes) must be positive")
    names = data.get("names", [])
    if len(names) != data.get("nc", 0):
        errors.append(f"'names' length ({len(names)}) should equal nc ({data.get('nc')})")

    return len(errors) == 0, errors
