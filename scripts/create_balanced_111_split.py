from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p).resolve()


def resolve_split_path(root: Path, dataset_root: Path, split_value: str) -> Path:
    p = Path(split_value)
    if p.is_absolute():
        return p
    candidate_dataset = (dataset_root / p).resolve()
    if candidate_dataset.exists():
        return candidate_dataset
    return (root / p).resolve()


def image_to_label(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def parse_classes(label_path: Path) -> List[int]:
    if not label_path.exists():
        return []
    classes: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except ValueError:
            continue
    return classes


def assign_primary_class(classes: List[int], priority: List[int]) -> int | None:
    class_set = set(classes)
    for cid in priority:
        if cid in class_set:
            return cid
    return None


def build_balanced_list(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    target_per_class: int,
    seed: int,
) -> Tuple[List[Path], Dict[int, int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Path]] = defaultdict(list)
    # Prioritize minority classes when an image contains multiple classes.
    class_priority = list(range(1, num_classes)) + [0]

    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
    for img in images:
        classes = parse_classes(image_to_label(img, labels_dir))
        if not classes:
            continue
        primary = assign_primary_class(classes, class_priority)
        if primary is None:
            continue
        by_class[primary].append(img.resolve())

    balanced: List[Path] = []
    summary: Dict[int, int] = {}
    for cid in range(num_classes):
        pool = by_class.get(cid, [])
        if not pool:
            summary[cid] = 0
            continue
        if len(pool) >= target_per_class:
            chosen = rng.sample(pool, target_per_class)
        else:
            chosen = list(pool)
            chosen.extend(rng.choices(pool, k=target_per_class - len(pool)))
        balanced.extend(chosen)
        summary[cid] = len(chosen)

    rng.shuffle(balanced)
    return balanced, summary


def write_list(path: Path, items: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in items) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict 1:1:1 balanced split lists")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--output-dir", default="dataset/balanced_111")
    parser.add_argument("--train-target", type=int, default=21)
    parser.add_argument("--val-target", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_yaml_path = resolve_path(root, args.data)
    cfg = load_yaml(data_yaml_path)

    dataset_root = Path(cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()

    train_images = resolve_split_path(root, dataset_root, cfg["train"])
    val_images = resolve_split_path(root, dataset_root, cfg["val"])
    train_labels = train_images.parent.parent / "labels" / train_images.name
    val_labels = val_images.parent.parent / "labels" / val_images.name

    nc = int(cfg["nc"])
    names = cfg["names"]

    train_list, train_summary = build_balanced_list(
        images_dir=train_images,
        labels_dir=train_labels,
        num_classes=nc,
        target_per_class=args.train_target,
        seed=args.seed,
    )
    val_list, val_summary = build_balanced_list(
        images_dir=val_images,
        labels_dir=val_labels,
        num_classes=nc,
        target_per_class=args.val_target,
        seed=args.seed + 1,
    )

    out_dir = resolve_path(root, args.output_dir)
    train_txt = out_dir / "train_111.txt"
    val_txt = out_dir / "val_111.txt"
    out_yaml = out_dir / "data_111.yaml"

    write_list(train_txt, train_list)
    write_list(val_txt, val_list)

    new_cfg = {
        "path": str(root.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "test": "",
        "nc": nc,
        "names": names,
    }
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    named_train = {names[k]: v for k, v in sorted(train_summary.items())}
    named_val = {names[k]: v for k, v in sorted(val_summary.items())}
    print(f"Wrote: {out_yaml}")
    print(f"Train (primary-class) samples per class: {named_train}")
    print(f"Val (primary-class) samples per class: {named_val}")
    print(f"Train list size: {len(train_list)}")
    print(f"Val list size: {len(val_list)}")


if __name__ == "__main__":
    main()

