from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def safe(v: float) -> str:
    return f"{v:.4f}"


def to_percent(v: float) -> str:
    return f"{v * 100:.2f}"


def load_report(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def weakest_classes(class_rows: List[Dict], k: int = 2) -> List[Dict]:
    if not class_rows:
        return []
    return sorted(class_rows, key=lambda r: r.get("f1", 0.0))[:k]


def build_per_class_csv(class_rows: List[Dict]) -> str:
    lines = ["class,precision,recall,f1,mAP50,precision_pct,recall_pct,f1_pct,mAP50_pct"]
    for r in class_rows:
        p = float(r["precision"])
        rc = float(r["recall"])
        f1 = float(r["f1"])
        m50 = float(r["mAP50"])
        lines.append(
            f"{r['class']},{safe(p)},{safe(rc)},{safe(f1)},{safe(m50)},{to_percent(p)},{to_percent(rc)},{to_percent(f1)},{to_percent(m50)}"
        )
    return "\n".join(lines) + "\n"


def build_experiment_csv(experiments: List[Dict]) -> str:
    header = (
        "experiment,model,best_epoch,precision,recall,mAP50,mAP50_95,macro_precision,macro_recall,macro_f1,macro_mAP50,best_weights"
    )
    lines = [header]
    for e in experiments:
        o = e.get("overall", {})
        m = e.get("macro_avg", {})
        lines.append(
            ",".join(
                [
                    e.get("run_name", ""),
                    e.get("model", ""),
                    str(o.get("epoch", "")),
                    safe(float(o.get("precision", 0.0))),
                    safe(float(o.get("recall", 0.0))),
                    safe(float(o.get("mAP50", 0.0))),
                    safe(float(o.get("mAP50_95", 0.0))),
                    safe(float(m.get("precision", 0.0))),
                    safe(float(m.get("recall", 0.0))),
                    safe(float(m.get("f1", 0.0))),
                    safe(float(m.get("mAP50", 0.0))),
                    e.get("best_weights", ""),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def build_latex_table(class_rows: List[Dict], caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Class & Precision & Recall & F1-Score & mAP@0.5 \\\\",
        "\\hline",
    ]
    for r in class_rows:
        lines.append(
            f"{r['class'].replace('_', ' ').title()} & {to_percent(float(r['precision']))} & {to_percent(float(r['recall']))} & {to_percent(float(r['f1']))} & {to_percent(float(r['mAP50']))} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_markdown_summary(best: Dict, weak: List[Dict]) -> str:
    o = best.get("overall", {})
    m = best.get("macro_avg", {})
    lines = [
        "# Publication-style Results Summary",
        "",
        f"- Selected experiment: **{best.get('run_name', '')}** ({best.get('model', '')})",
        f"- Best checkpoint: `{best.get('best_weights', '')}`",
        f"- Overall metrics: Precision={safe(float(o.get('precision', 0.0)))}, Recall={safe(float(o.get('recall', 0.0)))}, mAP@0.5={safe(float(o.get('mAP50', 0.0)))}, mAP@0.5:0.95={safe(float(o.get('mAP50_95', 0.0)))}",
        f"- Macro-average: Precision={safe(float(m.get('precision', 0.0)))}, Recall={safe(float(m.get('recall', 0.0)))}, F1={safe(float(m.get('f1', 0.0)))}, mAP@0.5={safe(float(m.get('mAP50', 0.0)))}",
        "",
        "## Weakest Classes",
    ]
    if weak:
        for w in weak:
            lines.append(
                f"- {w['class']}: P={safe(float(w['precision']))}, R={safe(float(w['recall']))}, F1={safe(float(w['f1']))}, mAP50={safe(float(w['mAP50']))}"
            )
    else:
        lines.append("- Not available (no per-class rows found).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready tables from final_report.json.")
    parser.add_argument("--report-json", required=True, help="Path to final_report.json")
    parser.add_argument("--output-dir", default="results/publication", help="Directory to save publication outputs")
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    out_dir = Path(args.output_dir).resolve()
    report = load_report(report_path)

    experiments = report.get("experiments", [])
    best_name = report.get("best_experiment", "")
    best = next((e for e in experiments if e.get("run_name") == best_name), experiments[0] if experiments else {})
    class_rows = best.get("class_rows", [])
    weak = weakest_classes(class_rows, k=2)

    write_text(out_dir / "experiments_summary.csv", build_experiment_csv(experiments))
    write_text(out_dir / "per_class_metrics.csv", build_per_class_csv(class_rows))
    write_text(
        out_dir / "per_class_table.tex",
        build_latex_table(
            class_rows=class_rows,
            caption="Per-class performance of the selected YOLOv8 model on marine waste detection.",
            label="tab:per_class_results",
        ),
    )
    write_text(out_dir / "publication_summary.md", build_markdown_summary(best, weak))

    print(f"Saved: {out_dir / 'experiments_summary.csv'}")
    print(f"Saved: {out_dir / 'per_class_metrics.csv'}")
    print(f"Saved: {out_dir / 'per_class_table.tex'}")
    print(f"Saved: {out_dir / 'publication_summary.md'}")


if __name__ == "__main__":
    main()
