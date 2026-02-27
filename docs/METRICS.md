# Evaluation Metrics: mAP and IoU (Explained)

This document explains the detection metrics used in the project: **IoU**, **Precision**, **Recall**, **AP**, and **mAP**.

---

## IoU (Intersection over Union)

**What it is:** IoU measures how much two axis-aligned boxes overlap.

- **Formula:**  
  `IoU = (area of intersection) / (area of union)`

- **Range:** 0 (no overlap) to 1 (boxes identical).

- **Why it matters:**  
  A predicted box is usually counted as a **correct detection** (True Positive) only if its IoU with some ground-truth box of the same class is **at least a threshold** (e.g. **0.5** for mAP@0.5). IoU is also used inside the model (e.g. loss, NMS).

**In code:**  
`evaluation.metrics.iou_box(box1, box2, format="xyxy")`  
`evaluation.metrics.iou_matrix(boxes1, boxes2)` for many boxes at once.

---

## Precision

**Formula:**  
`Precision = TP / (TP + FP)`

- **TP (True Positives):** Predictions that correctly match a ground-truth object (same class, IoU ≥ threshold).
- **FP (False Positives):** Predictions that do not match any ground-truth (wrong or duplicate).

**Meaning:** *“Of all the detections the model made, how many were correct?”*  
High precision ⇒ few false alarms.

---

## Recall

**Formula:**  
`Recall = TP / (TP + FN)`

- **FN (False Negatives):** Ground-truth objects that no prediction matched.

**Meaning:** *“Of all the real objects, how many did the model find?”*  
High recall ⇒ few missed objects.

---

## AP (Average Precision)

**What it is:** For **one class**, AP summarizes how good the model is across all confidence levels.

1. Sort all predictions for that class by **confidence** (high to low).
2. At each step, compute **precision** and **recall** (cumulative TP/FP/FN).
3. **AP** = area under the **Precision–Recall curve** (or 11-point interpolation in PASCAL VOC).

**Range:** 0–1; higher is better.

**In code:**  
`compute_ap(...)` with `use_11_point=True` for 11-point, or `False` for COCO-style (all-points) AP.

---

## mAP (mean Average Precision)

**What it is:**  
`mAP = mean(AP over all classes)`

- **mAP@0.5:** Use **IoU threshold 0.5** to decide if a prediction matches a ground truth. Single number; very common in papers.
- **mAP@0.5:0.95 (COCO):** Compute mAP at IoU thresholds **0.5, 0.55, 0.60, …, 0.95**, then **average** those mAP values. Stricter; used in COCO evaluation.

**In code:**  
- `compute_metrics(...)` returns `mAP_50` and `mAP_50_95`.  
- `compute_map(...)` for mAP at one IoU threshold.  
- `compute_map_coco_style(...)` for mAP@0.5:0.95.

---

## Summary

| Metric    | Formula / idea                         | Use |
|----------|-----------------------------------------|-----|
| **IoU**  | intersection / union of two boxes       | Match criterion (e.g. IoU ≥ 0.5 = TP). |
| **Precision** | TP / (TP + FP)                    | How many detections are correct. |
| **Recall**    | TP / (TP + FN)                    | How many ground truths were found. |
| **AP**   | Area under P–R curve (per class)        | Quality for one class. |
| **mAP@0.5**   | Mean of AP over classes at IoU=0.5   | Main detection score. |
| **mAP@0.5:0.95** | Mean of mAP at IoU 0.5–0.95      | Stricter COCO-style score. |

All of these are implemented in **`evaluation/metrics.py`** with docstrings; run `python -m evaluation.metrics` for a quick demo and plots.
