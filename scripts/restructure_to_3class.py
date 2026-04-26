from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# Old ids from current dataset/data.yaml:
# 0 plastic_bottle, 1 plastic_bag, 2 fishing_net, 3 can, 4 other_waste
OLD_TO_NEW = {
    0: 0,  # plastic_bottle -> plastic_waste
    1: 0,  # plastic_bag -> plastic_waste
    2: 2,  # fishing_net -> other_waste
    3: 1,  # can -> metal_waste
    4: 2,  # other_waste -> other_waste
}

NEW_NAMES = ["plastic_waste", "metal_waste", "other_waste"]


def read_label_rows(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return rows
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_old = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        # Clamp malformed normalized values.
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        rows.append((cls_old, x, y, w, h))
    return rows


def dedupe_rows(rows: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    seen = set()
    out = []
    for row in rows:
        key = (row[0], round(row[1], 6), round(row[2], 6), round(row[3], 6), round(row[4], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def rewrite_label_file(label_path: Path) -> Dict[str, int]:
    original = read_label_rows(label_path)
    remapped: List[Tuple[int, float, float, float, float]] = []
    dropped = 0
    for cls_old, x, y, w, h in original:
        if cls_old not in OLD_TO_NEW:
            dropped += 1
            continue
        cls_new = OLD_TO_NEW[cls_old]
        remapped.append((cls_new, x, y, w, h))
    deduped = dedupe_rows(remapped)
    duplicate_removed = len(remapped) - len(deduped)

    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in deduped]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "original_rows": len(original),
        "written_rows": len(deduped),
        "dropped_unknown": dropped,
        "duplicate_removed": duplicate_removed,
    }


def ensure_background_labels(images_dir: Path, labels_dir: Path) -> int:
    added = 0
    image_files = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    for img in image_files:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            lbl.write_text("", encoding="utf-8")
            added += 1
    return added


def count_boxes(labels_dir: Path) -> Counter:
    c = Counter()
    for lbl in labels_dir.glob("*.txt"):
        rows = read_label_rows(lbl)
        for cls, _, _, _, _ in rows:
            c[cls] += 1
    return c


def update_data_yaml(data_yaml: Path) -> None:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["nc"] = 3
    cfg["names"] = NEW_NAMES
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def resolve_split_path(project_root: Path, dataset_root: Path, split_value: str) -> Path:
    p = Path(split_value)
    if p.is_absolute():
        return p.resolve()
    c1 = (dataset_root / p).resolve()
    if c1.exists():
        return c1
    c2 = (project_root / p).resolve()
    if c2.exists():
        return c2
    return c1


def main() -> None:
    parser = argparse.ArgumentParser(description="Restructure YOLO labels from 5 classes to 3 classes and clean labels.")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--add-background-empty-labels", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_yaml = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    dataset_root = Path(cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    train_images = resolve_split_path(root, dataset_root, cfg["train"])
    val_images = resolve_split_path(root, dataset_root, cfg["val"])
    train_labels = train_images.parent.parent / "labels" / train_images.name
    val_labels = val_images.parent.parent / "labels" / val_images.name

    summary = {
        "train": {"files": 0, "rows_in": 0, "rows_out": 0, "dropped_unknown": 0, "duplicate_removed": 0},
        "val": {"files": 0, "rows_in": 0, "rows_out": 0, "dropped_unknown": 0, "duplicate_removed": 0},
        "background_labels_added": {"train": 0, "val": 0},
    }

    for split_name, labels_dir in [("train", train_labels), ("val", val_labels)]:
        for lbl in labels_dir.glob("*.txt"):
            stats = rewrite_label_file(lbl)
            summary[split_name]["files"] += 1
            summary[split_name]["rows_in"] += stats["original_rows"]
            summary[split_name]["rows_out"] += stats["written_rows"]
            summary[split_name]["dropped_unknown"] += stats["dropped_unknown"]
            summary[split_name]["duplicate_removed"] += stats["duplicate_removed"]

    if args.add_background_empty_labels:
        summary["background_labels_added"]["train"] = ensure_background_labels(train_images, train_labels)
        summary["background_labels_added"]["val"] = ensure_background_labels(val_images, val_labels)

    update_data_yaml(data_yaml)
    train_counts = count_boxes(train_labels)
    val_counts = count_boxes(val_labels)
    summary["class_counts"] = {
        "train": {NEW_NAMES[i]: int(train_counts[i]) for i in range(3)},
        "val": {NEW_NAMES[i]: int(val_counts[i]) for i in range(3)},
    }

    report_path = root / "dataset" / "restructure_report_3class.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    classes_txt = root / "classes.txt"
    classes_txt.write_text("\n".join(NEW_NAMES) + "\n", encoding="utf-8")

    print(f"Updated data yaml: {data_yaml}")
    print(f"Wrote class file: {classes_txt}")
    print(f"Report: {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
