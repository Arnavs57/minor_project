"""
Evaluation module for marine waste detection metrics.

Provides IoU, precision, recall, AP, and mAP with explanations
(see evaluation/metrics.py module docstring).
"""

from .metrics import (
    iou_box,
    iou_matrix,
    box_xyxy_to_xywh,
    box_xywh_to_xyxy,
    match_detections_to_ground_truth,
    precision_recall_from_tp_fp_fn,
    average_precision_11point,
    average_precision_all_points,
    compute_precision_recall_curve,
    compute_ap,
    compute_map,
    compute_map_coco_style,
    compute_metrics,
    plot_metrics,
    plot_iou_explanation,
)

__all__ = [
    "iou_box",
    "iou_matrix",
    "box_xyxy_to_xywh",
    "box_xywh_to_xyxy",
    "match_detections_to_ground_truth",
    "precision_recall_from_tp_fp_fn",
    "average_precision_11point",
    "average_precision_all_points",
    "compute_precision_recall_curve",
    "compute_ap",
    "compute_map",
    "compute_map_coco_style",
    "compute_metrics",
    "plot_metrics",
    "plot_iou_explanation",
]