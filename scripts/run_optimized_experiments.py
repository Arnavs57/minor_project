from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from ultralytics import YOLO


def read_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2 * precision * recall / denom


def parse_class_metrics_csv(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # "all" aggregate row is not per-class.
            cls_name = r.get("Class", "")
            if not cls_name or cls_name.lower() == "all":
                continue
            p = float(r.get("Box(P)", 0.0))
            rc = float(r.get("R", 0.0))
            map50 = float(r.get("mAP50", 0.0))
            rows.append(
                {
                    "class": cls_name,
                    "precision": p,
                    "recall": rc,
                    "f1": f1_score(p, rc),
                    "mAP50": map50,
                }
            )
    return rows


def class_rows_from_metrics(metrics) -> List[Dict[str, float]]:
    names_map = metrics.names if hasattr(metrics, "names") else {}
    p = list(metrics.box.p) if hasattr(metrics.box, "p") else []
    r = list(metrics.box.r) if hasattr(metrics.box, "r") else []
    m50 = list(metrics.box.ap50) if hasattr(metrics.box, "ap50") else []

    class_rows: List[Dict[str, float]] = []
    for idx in range(min(len(p), len(r), len(m50))):
        precision = float(p[idx])
        recall = float(r[idx])
        class_rows.append(
            {
                "class": str(names_map.get(idx, idx)),
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
                "mAP50": float(m50[idx]),
            }
        )
    return class_rows


def read_overall_metrics(results_csv: Path) -> Dict[str, float]:
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0.0)))
    return {
        "epoch": int(float(best.get("epoch", 0))),
        "precision": float(best.get("metrics/precision(B)", 0.0)),
        "recall": float(best.get("metrics/recall(B)", 0.0)),
        "mAP50": float(best.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(best.get("metrics/mAP50-95(B)", 0.0)),
    }


def macro_average(class_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not class_rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP50": 0.0}
    n = len(class_rows)
    return {
        "precision": sum(r["precision"] for r in class_rows) / n,
        "recall": sum(r["recall"] for r in class_rows) / n,
        "f1": sum(r["f1"] for r in class_rows) / n,
        "mAP50": sum(r["mAP50"] for r in class_rows) / n,
    }


def tune_thresholds(
    model_path: Path,
    data_yaml: Path,
    conf_candidates: List[float],
    iou_candidates: List[float],
    project_dir: Path,
    run_name: str,
) -> Dict[str, float]:
    model = YOLO(str(model_path))
    best = {"score": -1.0, "conf": 0.5, "iou": 0.5, "precision": 0.0, "recall": 0.0, "mAP50": 0.0}

    for conf in conf_candidates:
        for iou in iou_candidates:
            metrics = model.val(
                data=str(data_yaml),
                conf=conf,
                iou=iou,
                imgsz=640,
                project=str(project_dir / "threshold_tuning"),
                name=f"{run_name}_conf{conf:.2f}_iou{iou:.2f}",
                save_json=False,
                plots=False,
                verbose=False,
            )
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            f1 = f1_score(precision, recall)

            if f1 > best["score"]:
                best = {
                    "score": f1,
                    "conf": conf,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "mAP50": map50,
                }
    return best


def run_experiment(
    data_yaml: Path,
    model_name: str,
    run_name: str,
    out_project: Path,
    epochs: int,
    patience: int,
    batch: int,
    imgsz: int,
    mosaic: float,
    mixup: float,
    cos_lr: bool,
    extra_train_args: Dict[str, float],
) -> Dict:
    model = YOLO(model_name)
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "patience": patience,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(out_project),
        "name": run_name,
        "pretrained": True,
        "cos_lr": cos_lr,
        "optimizer": "auto",
        "mosaic": mosaic,
        "mixup": mixup,
        "save": True,
        "save_period": -1,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }
    train_args.update(extra_train_args)
    model.train(**train_args)

    run_dir = out_project / run_name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    # Keep only best model as requested.
    if last_pt.exists():
        last_pt.unlink()

    # Ultralytics validation requires a positive integer batch size.
    val_batch = batch if isinstance(batch, int) and batch > 0 else 2
    val_metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=val_batch,
        conf=0.25,
        iou=0.7,
        project=str(run_dir),
        name="val_baseline",
        plots=True,
        save_json=False,
        verbose=True,
    )

    class_csv = run_dir / "val_baseline" / "class_metrics.csv"
    class_rows = class_rows_from_metrics(val_metrics)
    if not class_rows and class_csv.exists():
        class_rows = parse_class_metrics_csv(class_csv)
    macro = macro_average(class_rows)

    results_csv = run_dir / "results.csv"
    overall = read_overall_metrics(results_csv) if results_csv.exists() else {
        "epoch": 0,
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "mAP50": float(val_metrics.box.map50),
        "mAP50_95": float(val_metrics.box.map),
    }

    return {
        "run_name": run_name,
        "model": model_name,
        "run_dir": str(run_dir),
        "best_weights": str(best_pt),
        "overall": overall,
        "macro_avg": macro,
        "class_rows": class_rows,
        "failure_cases": [],
    }


def print_table(rows: List[Dict[str, float]]) -> str:
    if not rows:
        return "No per-class rows available."
    headers = ["class", "precision", "recall", "f1", "mAP50"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for r in rows:
        lines.append(
            f"{r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['mAP50']:.4f}"
        )
    return "\n".join(lines)


def select_best(experiments: List[Dict]) -> Dict:
    return max(
        experiments,
        key=lambda e: (
            e["macro_avg"]["f1"],
            e["macro_avg"]["mAP50"],
            e["overall"]["mAP50"],
        ),
    )


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {"completed": [], "results": []}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"completed": [], "results": []}


def save_state(state_path: Path, completed: List[str], results: List[Dict]) -> None:
    state = {"completed": completed, "results": results}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        # Cross-platform non-destructive liveness check.
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path: Path) -> None:
    current_pid = os.getpid()
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_pid = int(lock_data.get("pid", -1))
        except (json.JSONDecodeError, ValueError, TypeError):
            lock_pid = -1
        if _pid_alive(lock_pid):
            raise RuntimeError(
                f"Another optimized run is active (pid={lock_pid}). "
                f"Stop it before starting a new run. Lock: {lock_path}"
            )
        # Stale lock from dead process: safe to replace.
    lock_payload = {"pid": current_pid}
    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")


def release_run_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_pid = int(lock_data.get("pid", -1))
    except (json.JSONDecodeError, ValueError, TypeError):
        lock_pid = -1
    if lock_pid == os.getpid():
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-efficient YOLOv8 experiments and produce report-ready metrics.")
    parser.add_argument("--data", default="dataset/data.yaml", help="YOLO data yaml path")
    parser.add_argument("--project", default="runs/optimized", help="Output project directory")
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs (must be <= 100)")
    parser.add_argument("--patience", type=int, default=15, help="Early stop patience")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 auto max GPU)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.15)
    parser.add_argument("--skip-threshold-tuning", action="store_true")
    parser.add_argument("--include-exp-c", action="store_true", help="Run optional experiment C")
    parser.add_argument("--exp-c-imgsz", type=int, default=768, help="Image size for optional experiment C")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state file")
    parser.add_argument(
        "--pause-after",
        choices=["a", "b", "c"],
        default=None,
        help="Pause pipeline after this experiment label",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to pipeline state file (default: <project>/pipeline_state.json)",
    )
    args = parser.parse_args()

    if args.epochs > 100:
        raise ValueError("epochs must be <= 100 as per optimization rules")

    root = Path(__file__).resolve().parent.parent
    data_yaml = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    project_dir = (root / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    lock_path = project_dir / "run.lock.json"
    state_path = (
        (root / args.state_file).resolve()
        if args.state_file and not Path(args.state_file).is_absolute()
        else (Path(args.state_file) if args.state_file else (project_dir / "pipeline_state.json"))
    )
    acquire_run_lock(lock_path)

    try:
        _ = read_data_yaml(data_yaml)

        experiments_cfg: List[Dict] = [
            {
                "model": "yolov8s.pt",
                "name": "exp_a_yolov8s",
                "epochs": 60,
                "patience": 15,
                "imgsz": 640,
                "extra": {},
            },
            {
                "model": "yolov8m.pt",
                "name": "exp_b_yolov8m",
                "epochs": 80,
                "patience": 15,
                "imgsz": 640,
                "extra": {},
            },
        ]
        if args.include_exp_c:
            experiments_cfg.append(
                {
                    "model": "yolov8m.pt",
                    "name": f"exp_c_yolov8m_img{args.exp_c_imgsz}",
                    "epochs": 80,
                    "patience": 15,
                    "imgsz": args.exp_c_imgsz,
                    "extra": {
                        "lr0": 0.006,
                        "lrf": 0.01,
                        "weight_decay": 0.0007,
                        "warmup_epochs": 3.0,
                        "box": 7.5,
                        "cls": 0.6,
                        "dfl": 1.8,
                    },
                }
            )

        if args.pause_after == "c" and not args.include_exp_c:
            raise ValueError("--pause-after c requires --include-exp-c")

        exp_label_by_name = {
            "exp_a_yolov8s": "a",
            "exp_b_yolov8m": "b",
        }
        if args.include_exp_c:
            exp_label_by_name[f"exp_c_yolov8m_img{args.exp_c_imgsz}"] = "c"

        completed: List[str] = []
        all_results: List[Dict] = []
        if args.resume:
            state = load_state(state_path)
            completed = list(state.get("completed", []))
            all_results = list(state.get("results", []))
            print(f"Resuming from state: {state_path}")
            print(f"Already completed: {completed}")

        for exp in experiments_cfg:
            if exp["name"] in completed:
                print(f"Skipping completed experiment: {exp['name']}")
                continue

            result = run_experiment(
                data_yaml=data_yaml,
                model_name=exp["model"],
                run_name=exp["name"],
                out_project=project_dir,
                epochs=exp["epochs"],
                patience=exp["patience"],
                batch=args.batch,
                imgsz=exp["imgsz"],
                mosaic=args.mosaic,
                mixup=args.mixup,
                cos_lr=True,
                extra_train_args=exp["extra"],
            )

            if not args.skip_threshold_tuning and Path(result["best_weights"]).exists():
                tuned = tune_thresholds(
                    model_path=Path(result["best_weights"]),
                    data_yaml=data_yaml,
                    conf_candidates=[0.50, 0.55, 0.60],
                    iou_candidates=[0.50],
                    project_dir=project_dir,
                    run_name=exp["name"],
                )
                result["threshold_tuning"] = tuned
                result["threshold_compare"] = {
                    "before": {
                        "precision": result["overall"]["precision"],
                        "recall": result["overall"]["recall"],
                        "mAP50": result["overall"]["mAP50"],
                        "f1": f1_score(result["overall"]["precision"], result["overall"]["recall"]),
                    },
                    "after": {
                        "precision": tuned["precision"],
                        "recall": tuned["recall"],
                        "mAP50": tuned["mAP50"],
                        "f1": tuned["score"],
                        "conf": tuned["conf"],
                        "iou": tuned["iou"],
                    },
                }
            macro_f1 = result["macro_avg"]["f1"]
            result["failure_cases"] = [
                r["class"] for r in result["class_rows"] if r["f1"] < macro_f1
            ]
            all_results.append(result)
            completed.append(exp["name"])
            save_state(state_path, completed, all_results)

            exp_label = exp_label_by_name.get(exp["name"])
            if args.pause_after and exp_label == args.pause_after:
                print(f"Paused after experiment {exp_label.upper()} ({exp['name']}).")
                print(f"Resume with: --resume --state-file \"{state_path}\"")
                return

        if not all_results:
            raise RuntimeError("No experiment results available. Check state file and run options.")

        best = select_best(all_results)
        weak_classes = sorted(best["class_rows"], key=lambda r: r["f1"])[:2] if best["class_rows"] else []

        report = {
            "selection_rule": "Max macro F1, then macro mAP50, then overall mAP50",
            "best_experiment": best["run_name"],
            "best_model": best["model"],
            "best_weights": best["best_weights"],
            "experiments": all_results,
            "weakest_classes": weak_classes,
        }
        report_path = project_dir / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        md_lines: List[str] = []
        md_lines.append("# Optimized YOLOv8 Experiment Report")
        md_lines.append("")
        md_lines.append(f"- Best experiment: **{best['run_name']}** ({best['model']})")
        md_lines.append(f"- Best weights: `{best['best_weights']}`")
        md_lines.append(
            f"- Overall: mAP@0.5={best['overall']['mAP50']:.4f}, "
            f"Precision={best['overall']['precision']:.4f}, Recall={best['overall']['recall']:.4f}"
        )
        md_lines.append(
            f"- Macro avg: F1={best['macro_avg']['f1']:.4f}, mAP@0.5={best['macro_avg']['mAP50']:.4f}"
        )
        if "threshold_tuning" in best:
            t = best["threshold_tuning"]
            md_lines.append(
                f"- Best thresholds: conf={t['conf']:.2f}, iou={t['iou']:.2f} "
                f"(Precision={t['precision']:.4f}, Recall={t['recall']:.4f}, F1={t['score']:.4f})"
            )
        if "threshold_compare" in best:
            b = best["threshold_compare"]["before"]
            a = best["threshold_compare"]["after"]
            md_lines.append(
                f"- Before/After tuning F1: {b['f1']:.4f} -> {a['f1']:.4f}, "
                f"Precision: {b['precision']:.4f} -> {a['precision']:.4f}"
            )
        md_lines.append("")
        md_lines.append("## Per-class Performance (Best Run)")
        md_lines.append("")
        md_lines.append(print_table(best["class_rows"]))
        md_lines.append("")
        md_lines.append("## Weakest Classes")
        md_lines.append("")
        if weak_classes:
            for w in weak_classes:
                md_lines.append(
                    f"- {w['class']}: P={w['precision']:.4f}, R={w['recall']:.4f}, F1={w['f1']:.4f}, mAP50={w['mAP50']:.4f}"
                )
        else:
            md_lines.append("- No per-class entries found.")
        md_lines.append("")
        md_lines.append("## Failure Case Classes")
        md_lines.append("")
        if best.get("failure_cases"):
            for c in best["failure_cases"]:
                md_lines.append(f"- {c}")
        else:
            md_lines.append("- None identified from macro-F1 criterion.")

        report_md = project_dir / "final_report.md"
        report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        print(f"Saved JSON report: {report_path}")
        print(f"Saved Markdown report: {report_md}")
        print(f"Selected best experiment: {best['run_name']}")
        print(f"Best model weights: {best['best_weights']}")
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    main()
