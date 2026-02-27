"""
Evaluation metrics for object detection: IoU, mAP, precision, recall.

--------------------------------------------------------------------------------
METRIC EXPLANATIONS
--------------------------------------------------------------------------------

IoU (Intersection over Union)
-----------------------------
    IoU measures overlap between a predicted box and a ground-truth box:
        IoU = (area of intersection) / (area of union)
    Range: 0 (no overlap) to 1 (perfect match).
    A prediction is considered correct (True Positive) if IoU >= threshold
    (e.g. 0.5 for mAP@0.5). IoU is also used in loss functions and NMS.

Precision
---------
    Precision = TP / (TP + FP)
    "Of all detections the model made, how many were correct?"
    High precision = few false alarms.

Recall
------
    Recall = TP / (TP + FN)
    "Of all ground-truth objects, how many did the model find?"
    High recall = few missed objects.

AP (Average Precision)
----------------------
    For one class: sort predictions by confidence (descending), then compute
    precision and recall at each step. AP is the area under the Precision-Recall
    curve. Often computed with 11-point interpolation (PASCAL) or all-points
    (COCO). Unit: 0–1; higher is better.

mAP (mean Average Precision)
----------------------------
    mAP = mean(AP over all classes).
    - mAP@0.5: use IoU threshold 0.5 to decide TP (one number).
    - mAP@0.5:0.95 (COCO): average of mAP at IoU thresholds 0.5, 0.55, ..., 0.95.
    Standard single-number metric for detection; higher is better.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional matplotlib/seaborn for plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# -----------------------------------------------------------------------------
# IoU (Intersection over Union)
# -----------------------------------------------------------------------------


def box_xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert one box [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return np.array([xc, yc, w, h], dtype=box.dtype)


def box_xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert one box [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    xc, yc, w, h = box[:4]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.array([x1, y1, x2, y2], dtype=box.dtype)


def iou_box(
    box1: np.ndarray,
    box2: np.ndarray,
    format: str = "xyxy",
) -> float:
    """
    IoU (Intersection over Union) between two boxes.

    IoU = (intersection area) / (union area). Used to decide if a detection
    matches a ground truth (e.g. match if IoU >= 0.5).

    Args:
        box1: [x1, y1, x2, y2] or [xc, yc, w, h], shape (4,).
        box2: Same format as box1.
        format: "xyxy" (x1,y1,x2,y2) or "xywh" (center, w, h).

    Returns:
        IoU in [0, 1]. 0 = no overlap, 1 = identical boxes.
    """
    b1 = np.asarray(box1, dtype=np.float64)
    b2 = np.asarray(box2, dtype=np.float64)
    if format == "xywh":
        b1 = box_xywh_to_xyxy(b1)
        b2 = box_xywh_to_xyxy(b2)
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    format: str = "xyxy",
) -> np.ndarray:
    """
    Pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) in xyxy or xywh.
        boxes2: (M, 4) in same format.
        format: "xyxy" or "xywh".

    Returns:
        (N, M) matrix of IoU values.
    """
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    if format == "xywh":
        boxes1 = np.array([box_xywh_to_xyxy(b[:4]) for b in boxes1])
        boxes2 = np.array([box_xywh_to_xyxy(b[:4]) for b in boxes2])
    # (N,4) and (M,4) -> intersection
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union


# -----------------------------------------------------------------------------
# Matching and TP / FP / FN
# -----------------------------------------------------------------------------


def match_detections_to_ground_truth(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    box_format: str = "xyxy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth by IoU (greedy, best IoU first).

    Each ground-truth object is matched to at most one prediction (highest IoU
    above threshold, same class). Unmatched predictions = FP, unmatched GT = FN.

    Args:
        pred_boxes: (N, 4) predicted boxes.
        pred_classes: (N,) integer class ids.
        pred_scores: (N,) confidence scores (used for ordering).
        gt_boxes: (M, 4) ground-truth boxes.
        gt_classes: (M,) integer class ids.
        iou_threshold: Minimum IoU to count as a match.
        box_format: "xyxy" or "xywh".

    Returns:
        tp: boolean array (N,) True where prediction is a TP.
        fp: boolean array (N,) True where prediction is a FP.
        fn: boolean array (M,) True where GT has no matching prediction.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    fn = np.ones(n_gt, dtype=bool)  # initially all GT unmatched

    if n_pred == 0:
        return tp, np.zeros(n_pred, dtype=bool), fn
    if n_gt == 0:
        return tp, np.ones(n_pred, dtype=bool), np.zeros(0, dtype=bool)

    # Sort predictions by confidence (descending)
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_classes = pred_classes[order]

    iou_mat = iou_matrix(pred_boxes, gt_boxes, format=box_format)  # (N, M)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for i in range(n_pred):
        c = pred_classes[i]
        # Only consider GT of same class
        same_class = gt_classes == c
        ious = iou_mat[i, :].copy()
        ious[~same_class] = 0
        j = np.argmax(ious)
        if ious[j] >= iou_threshold and not gt_matched[j]:
            gt_matched[j] = True
            tp[order[i]] = True

    fn = ~gt_matched
    fp = ~tp
    return tp, fp, fn


# -----------------------------------------------------------------------------
# Precision, Recall, AP, mAP
# -----------------------------------------------------------------------------


def precision_recall_from_tp_fp_fn(
    tp: Union[int, np.ndarray],
    fp: Union[int, np.ndarray],
    fn: Union[int, np.ndarray],
) -> Tuple[float, float]:
    """
    Precision = TP / (TP + FP), Recall = TP / (TP + FN).

    When TP+FP=0, precision is 1 by convention. When TP+FN=0, recall is 0.
    """
    tp, fp, fn = float(np.sum(tp)), float(np.sum(fp)), float(np.sum(fn))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def average_precision_11point(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """
    PASCAL VOC 11-point interpolation: AP = (1/11) * sum(max prec at r in [0,0.1,...,1]).

    precisions and recalls should be aligned (e.g. at each confidence threshold).
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    t = np.linspace(0, 1, 11)
    ap = 0.0
    for r in t:
        p = precisions[recalls >= r]
        ap += np.max(p) if len(p) > 0 else 0.0
    return ap / 11.0


def average_precision_all_points(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """
    COCO-style AP: area under Precision-Recall curve (recall increasing).

    Sorts by recall and computes area under curve (trapezoidal).
    """
    if len(precisions) < 2 or len(recalls) < 2:
        return float(precisions[0]) if len(precisions) else 0.0
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    # Make precision monotonically decreasing in recall
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    try:
        return float(np.trapezoid(precisions, recalls))
    except AttributeError:
        return float(np.trapz(precisions, recalls))


def compute_precision_recall_curve(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    box_format: str = "xyxy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall at each confidence threshold (sorted descending).

    Returns:
        precisions: array of precision values (one per threshold).
        recalls: array of recall values (one per threshold).
    """
    if len(pred_scores) == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_classes = pred_classes[order]
    pred_scores = pred_scores[order]

    n_gt = len(gt_boxes)
    tp_cum = np.zeros(len(pred_boxes) + 1)
    fp_cum = np.zeros(len(pred_boxes) + 1)
    for k in range(1, len(pred_boxes) + 1):
        tp, fp, fn = match_detections_to_ground_truth(
            pred_boxes[:k],
            pred_classes[:k],
            pred_scores[:k],
            gt_boxes,
            gt_classes,
            iou_threshold=iou_threshold,
            box_format=box_format,
        )
        tp_cum[k] = np.sum(tp)
        fp_cum[k] = np.sum(fp)
    tp_cum = tp_cum[1:]
    fp_cum = fp_cum[1:]
    fn_cum = n_gt - tp_cum
    precisions = np.where(tp_cum + fp_cum > 0, tp_cum / (tp_cum + fp_cum), 1.0)
    recalls = np.where(n_gt > 0, tp_cum / n_gt, 0.0)
    return precisions, recalls


def compute_ap(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    use_11_point: bool = False,
    box_format: str = "xyxy",
) -> float:
    """
    Average Precision for one class (if single class) or all classes mixed.

    For per-class AP, filter pred_* and gt_* by class first, then call this.
    """
    prec, rec = compute_precision_recall_curve(
        pred_boxes, pred_classes, pred_scores,
        gt_boxes, gt_classes,
        iou_threshold=iou_threshold,
        box_format=box_format,
    )
    if use_11_point:
        return average_precision_11point(prec, rec)
    return average_precision_all_points(prec, rec)


def compute_map(
    all_pred_boxes: List[np.ndarray],
    all_pred_classes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_classes: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
    use_11_point: bool = False,
    box_format: str = "xyxy",
) -> Tuple[float, Dict[int, float]]:
    """
    mAP = mean of AP over all classes.

    Args:
        all_*: Lists of arrays (one per image) for a dataset.
        num_classes: Number of classes (class ids in [0, num_classes-1]).
        iou_threshold: IoU threshold for matching (e.g. 0.5 for mAP@0.5).
        use_11_point: Use 11-point interpolation for AP.
        box_format: "xyxy" or "xywh".

    Returns:
        mAP: scalar.
        ap_per_class: dict class_id -> AP (only classes with GT).
    """
    ap_per_class: Dict[int, float] = {}
    for c in range(num_classes):
        pred_boxes_c = []
        pred_scores_c = []
        gt_boxes_c = []
        for i in range(len(all_gt_boxes)):
            gt_mask = all_gt_classes[i] == c
            pred_mask = all_pred_classes[i] == c
            if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                continue
            gt_boxes_c.append(all_gt_boxes[i][gt_mask])
            pred_boxes_c.append(all_pred_boxes[i][pred_mask])
            pred_scores_c.append(all_pred_scores[i][pred_mask])
        # Flatten to one set per class across images
        n_gt_c = sum(len(x) for x in gt_boxes_c)
        if n_gt_c == 0:
            continue
        gt_flat = np.vstack([x for x in gt_boxes_c if len(x) > 0])
        pred_flat = np.vstack([x for x in pred_boxes_c if len(x) > 0]) if sum(len(x) for x in pred_boxes_c) > 0 else np.zeros((0, 4))
        scores_flat = np.concatenate([x for x in pred_scores_c if len(x) > 0]) if sum(len(x) for x in pred_scores_c) > 0 else np.zeros(0)
        gt_classes_flat = np.full(gt_flat.shape[0], c)
        pred_classes_flat = np.full(pred_flat.shape[0], c)
        if gt_flat.shape[0] == 0:
            ap_per_class[c] = 0.0
            continue
        ap_per_class[c] = compute_ap(
            pred_flat, pred_classes_flat, scores_flat,
            gt_flat, gt_classes_flat,
            iou_threshold=iou_threshold,
            use_11_point=use_11_point,
            box_format=box_format,
        )
    if not ap_per_class:
        return 0.0, {}
    mAP = float(np.mean(list(ap_per_class.values())))
    return mAP, ap_per_class


def compute_map_coco_style(
    all_pred_boxes: List[np.ndarray],
    all_pred_classes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_classes: List[np.ndarray],
    num_classes: int,
    iou_thresholds: Optional[np.ndarray] = None,
    box_format: str = "xyxy",
) -> Tuple[float, Dict[float, float]]:
    """
    mAP@0.5:0.95 (COCO): average of mAP at IoU thresholds 0.5, 0.55, ..., 0.95.

    Returns:
        mAP_5095: single scalar (average over IoU thresholds).
        map_per_iou: dict iou_threshold -> mAP at that threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map_per_iou: Dict[float, float] = {}
    for iou in iou_thresholds:
        m, _ = compute_map(
            all_pred_boxes, all_pred_classes, all_pred_scores,
            all_gt_boxes, all_gt_classes, num_classes,
            iou_threshold=float(iou),
            use_11_point=False,
            box_format=box_format,
        )
        map_per_iou[float(iou)] = m
    mAP_5095 = float(np.mean(list(map_per_iou.values())))
    return mAP_5095, map_per_iou


# -----------------------------------------------------------------------------
# High-level: single-call metrics
# -----------------------------------------------------------------------------


def compute_metrics(
    all_pred_boxes: List[np.ndarray],
    all_pred_classes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_classes: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
    box_format: str = "xyxy",
) -> Dict:
    """
    Compute detection metrics: precision, recall, mAP@0.5, mAP@0.5:0.95, IoU.

    All box lists are per-image; boxes are (4,) in xyxy or xywh. Classes are int.

    Returns:
        dict with keys: precision, recall, mAP_50, mAP_50_95, ap_per_class,
        map_per_iou (for 0.5:0.95), and optional per-image counts.
    """
    # Aggregate TP, FP, FN at iou_threshold across all images
    total_tp = total_fp = total_fn = 0
    for i in range(len(all_gt_boxes)):
        tp, fp, fn = match_detections_to_ground_truth(
            all_pred_boxes[i], all_pred_classes[i], all_pred_scores[i],
            all_gt_boxes[i], all_gt_classes[i],
            iou_threshold=iou_threshold,
            box_format=box_format,
        )
        total_tp += np.sum(tp)
        total_fp += np.sum(fp)
        total_fn += np.sum(fn)
    precision, recall = precision_recall_from_tp_fp_fn(total_tp, total_fp, total_fn)

    mAP_50, ap_per_class = compute_map(
        all_pred_boxes, all_pred_classes, all_pred_scores,
        all_gt_boxes, all_gt_classes, num_classes,
        iou_threshold=iou_threshold,
        box_format=box_format,
    )
    mAP_50_95, map_per_iou = compute_map_coco_style(
        all_pred_boxes, all_pred_classes, all_pred_scores,
        all_gt_boxes, all_gt_classes, num_classes,
        box_format=box_format,
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "ap_per_class": ap_per_class,
        "map_per_iou": map_per_iou,
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_metrics(
    metrics: Dict,
    output_path: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot metrics (precision, recall, mAP, per-class AP) and save figure.

    Requires matplotlib (and optionally seaborn).
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plot_metrics. pip install matplotlib")
    if _HAS_SNS:
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scalar metrics
    ax = axes[0]
    scalars = {
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "mAP@0.5": metrics.get("mAP_50", 0),
        "mAP@0.5:0.95": metrics.get("mAP_50_95", 0),
    }
    names = list(scalars.keys())
    values = list(scalars.values())
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    bars = ax.bar(names, values, color=colors[: len(names)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Detection metrics")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    # Right: per-class AP
    ax = axes[1]
    ap_per_class = metrics.get("ap_per_class", {})
    if ap_per_class:
        classes = sorted(ap_per_class.keys())
        labels = [class_names[c] if class_names and c < len(class_names) else f"Class {c}" for c in classes]
        vals = [ap_per_class[c] for c in classes]
        ax.barh(labels, vals, color="steelblue", alpha=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("AP@0.5")
        ax.set_title("AP per class")
    else:
        ax.text(0.5, 0.5, "No per-class AP", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_iou_explanation(output_path: Optional[Union[str, Path]] = None) -> None:
    """
    Draw two boxes and their intersection/union to illustrate IoU (optional helper).
    """
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Box 1: (1,1) to (4,4), Box 2: (2.5,2.5) to (5,5)
    b1 = [1, 1, 4, 4]
    b2 = [2.5, 2.5, 5, 5]
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((b1[0], b1[1]), b1[2] - b1[0], b1[3] - b1[1], fill=False, edgecolor="blue", linewidth=2, label="Box A"))
    ax.add_patch(Rectangle((b2[0], b2[1]), b2[2] - b2[0], b2[3] - b2[1], fill=False, edgecolor="orange", linewidth=2, label="Box B"))
    inter_x1, inter_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    inter_x2, inter_y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    ax.add_patch(Rectangle((inter_x1, inter_y1), inter_x2 - inter_x1, inter_y2 - inter_y1, fill=True, facecolor="green", alpha=0.5, label="Intersection"))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("IoU = Intersection / Union")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate detection metrics (mAP, IoU, precision, recall)")
    parser.add_argument("--predictions", help="Path to predictions (not yet implemented: use compute_metrics in code)")
    parser.add_argument("--ground-truth", help="Path to ground truth")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Plot IoU explanation figure")
    args = parser.parse_args()

    if args.plot:
        out = Path(args.output) / "iou_explanation.png"
        Path(args.output).mkdir(parents=True, exist_ok=True)
        plot_iou_explanation(out)
        print(f"Saved IoU explanation to {out}")
        return

    # Demo: compute metrics on dummy data
    print("Running demo with dummy boxes to verify metrics...")
    pred = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])]
    pred_c = [np.array([0, 0])]
    pred_s = [np.array([0.9, 0.8])]
    gt = [np.array([[0.12, 0.12, 0.32, 0.32], [0.52, 0.52, 0.72, 0.72]])]
    gt_c = [np.array([0, 0])]
    m = compute_metrics(pred, pred_c, pred_s, gt, gt_c, num_classes=1, iou_threshold=0.5)
    print("Precision:", m["precision"], "Recall:", m["recall"], "mAP@0.5:", m["mAP_50"], "mAP@0.5:0.95:", m["mAP_50_95"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(m, output_path=out_dir / "metrics.png")
    print(f"Saved metrics plot to {out_dir / 'metrics.png'}")


if __name__ == "__main__":
    main()
