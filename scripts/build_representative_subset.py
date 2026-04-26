from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


def read_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_path(root: Path, dataset_root: Path, split_value: str) -> Path:
    p = Path(split_value)
    if p.is_absolute():
        return p
    candidate1 = dataset_root / p
    if candidate1.exists():
        return candidate1
    return root / p


def image_to_label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def parse_label_classes(label_path: Path) -> Set[int]:
    classes: Set[int] = set()
    if not label_path.exists():
        return classes
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return classes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            classes.add(int(float(parts[0])))
        except ValueError:
            continue
    return classes


def count_boxes(label_path: Path) -> Counter:
    c = Counter()
    if not label_path.exists():
        return c
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        c[cls] += 1
    return c


def load_images_and_labels(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Set[int], Counter]]:
    image_files = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )
    rows: List[Tuple[Path, Set[int], Counter]] = []
    for img in image_files:
        lbl = image_to_label_path(img, labels_dir)
        classes = parse_label_classes(lbl)
        box_counts = count_boxes(lbl)
        if classes:
            rows.append((img, classes, box_counts))
    return rows


def pick_subset(
    rows: List[Tuple[Path, Set[int], Counter]],
    ratio: float,
    num_classes: int,
    seed: int,
) -> List[Tuple[Path, Set[int], Counter]]:
    rng = random.Random(seed)
    rng.shuffle(rows)

    target_size = max(1, int(len(rows) * ratio))
    global_counts = Counter()
    for _, classes, _ in rows:
        for c in classes:
            global_counts[c] += 1

    target_per_class: Dict[int, int] = {}
    for c in range(num_classes):
        if global_counts[c] == 0:
            target_per_class[c] = 0
        else:
            target_per_class[c] = max(1, int(global_counts[c] * ratio))

    chosen: List[Tuple[Path, Set[int], Counter]] = []
    chosen_by_class = Counter()
    chosen_set = set()

    # First pass: ensure each class target is approached.
    for c in sorted(range(num_classes), key=lambda x: global_counts[x]):
        if global_counts[c] == 0:
            continue
        need = target_per_class[c] - chosen_by_class[c]
        if need <= 0:
            continue
        for row in rows:
            img = row[0]
            if img in chosen_set:
                continue
            if c in row[1]:
                chosen.append(row)
                chosen_set.add(img)
                for k in row[1]:
                    chosen_by_class[k] += 1
                need -= 1
                if need <= 0:
                    break

    # Second pass: fill remaining by class rarity score.
    if len(chosen) < target_size:
        rarity = {}
        for c in range(num_classes):
            rarity[c] = 1.0 / max(global_counts[c], 1)
        remaining = [r for r in rows if r[0] not in chosen_set]
        remaining.sort(key=lambda r: sum(rarity[c] for c in r[1]), reverse=True)
        needed = target_size - len(chosen)
        chosen.extend(remaining[:needed])

    return chosen[:target_size]


def build_balanced_train_list(
    train_images: List[Path],
    train_labels_dir: Path,
    num_classes: int,
    max_repeat: int,
) -> List[Path]:
    class_to_images: Dict[int, List[Path]] = defaultdict(list)
    class_counts = Counter()
    for img in train_images:
        label_path = image_to_label_path(img, train_labels_dir)
        classes = parse_label_classes(label_path)
        for c in classes:
            class_to_images[c].append(img)
            class_counts[c] += 1

    nonzero = [class_counts[c] for c in range(num_classes) if class_counts[c] > 0]
    if not nonzero:
        return train_images
    target = max(nonzero)

    repeated: List[Path] = list(train_images)
    for c in range(num_classes):
        current = class_counts[c]
        if current == 0:
            continue
        needed_factor = min(max_repeat, max(1, target // current))
        if needed_factor <= 1:
            continue
        imgs = class_to_images[c]
        repeated.extend(imgs * (needed_factor - 1))
    return repeated


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative YOLO subset with optional train oversampling list.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to source YOLO data.yaml")
    parser.add_argument("--subset-ratio", type=float, default=0.25, help="Subset ratio (0.2 to 0.3 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="dataset/subset", help="Where subset YAML/list files are written")
    parser.add_argument("--oversample", action="store_true", help="Create oversampled train list for class balancing")
    parser.add_argument("--max-repeat", type=int, default=4, help="Max repeat factor for minority class images")
    args = parser.parse_args()

    if not (0.1 <= args.subset_ratio <= 1.0):
        raise ValueError("subset-ratio must be within [0.1, 1.0]")

    root = Path(__file__).resolve().parent.parent
    src_yaml = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    cfg = read_data_yaml(src_yaml)

    dataset_root = Path(cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (src_yaml.parent / dataset_root).resolve()

    train_images_dir = resolve_split_path(root, dataset_root, cfg["train"])
    val_images_dir = resolve_split_path(root, dataset_root, cfg["val"])
    train_labels_dir = train_images_dir.parent.parent / "labels" / train_images_dir.name
    val_labels_dir = val_images_dir.parent.parent / "labels" / val_images_dir.name

    num_classes = int(cfg["nc"])
    class_names = cfg.get("names", [str(i) for i in range(num_classes)])

    train_rows = load_images_and_labels(train_images_dir, train_labels_dir)
    val_rows = load_images_and_labels(val_images_dir, val_labels_dir)
    subset_train = pick_subset(train_rows, args.subset_ratio, num_classes, args.seed)
    subset_val = pick_subset(val_rows, args.subset_ratio, num_classes, args.seed + 1)

    out_dir = (root / args.output_dir).resolve()
    train_txt = out_dir / "train_subset.txt"
    val_txt = out_dir / "val_subset.txt"
    train_balanced_txt = out_dir / "train_subset_balanced.txt"
    yaml_out = out_dir / "data_subset.yaml"

    train_images = [row[0].resolve() for row in subset_train]
    val_images = [row[0].resolve() for row in subset_val]
    write_lines(train_txt, [str(p) for p in train_images])
    write_lines(val_txt, [str(p) for p in val_images])

    balanced_train_images = train_images
    if args.oversample:
        balanced_train_images = build_balanced_train_list(
            train_images=train_images,
            train_labels_dir=train_labels_dir,
            num_classes=num_classes,
            max_repeat=args.max_repeat,
        )
        write_lines(train_balanced_txt, [str(p) for p in balanced_train_images])

    train_for_yaml = train_balanced_txt if args.oversample else train_txt
    subset_yaml = {
        "path": str(root.resolve()),
        "train": str(train_for_yaml.resolve()),
        "val": str(val_txt.resolve()),
        "test": cfg.get("test", ""),
        "nc": num_classes,
        "names": class_names,
    }
    with yaml_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(subset_yaml, f, sort_keys=False)

    def summarize(rows: List[Tuple[Path, Set[int], Counter]]) -> Dict[str, object]:
        img_count = len(rows)
        class_img_presence = Counter()
        class_box_counts = Counter()
        for _, classes, box_c in rows:
            for c in classes:
                class_img_presence[c] += 1
            class_box_counts.update(box_c)
        return {
            "images": img_count,
            "class_image_presence": {class_names[k]: class_img_presence[k] for k in range(num_classes)},
            "class_box_counts": {class_names[k]: class_box_counts[k] for k in range(num_classes)},
        }

    report = {
        "source_data_yaml": str(src_yaml),
        "subset_ratio": args.subset_ratio,
        "oversample": args.oversample,
        "max_repeat": args.max_repeat if args.oversample else 1,
        "train_summary": summarize(subset_train),
        "val_summary": summarize(subset_val),
        "balanced_train_list_size": len(balanced_train_images),
        "output_yaml": str(yaml_out),
    }
    report_path = out_dir / "subset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Subset YAML: {yaml_out}")
    print(f"Subset report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
