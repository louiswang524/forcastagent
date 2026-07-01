from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((value - m) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def load_metrics(paths: list[Path]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for system, row in metrics.items():
            for metric, value in row.items():
                grouped[system][metric].append(float(value))
    return grouped


def write_summary(grouped: dict[str, dict[str, list[float]]], out_dir: Path, paths: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "system",
        "runs",
        "n_mean",
        "brier_mean",
        "brier_sd",
        "log_score_mean",
        "log_score_sd",
        "calibration_error_mean",
        "calibration_error_sd",
    ]
    rows = []
    for system in sorted(grouped):
        metrics = grouped[system]
        rows.append(
            {
                "system": system,
                "runs": len(metrics["brier"]),
                "n_mean": f"{mean(metrics['n']):.1f}",
                "brier_mean": f"{mean(metrics['brier']):.6f}",
                "brier_sd": f"{stdev(metrics['brier']):.6f}",
                "log_score_mean": f"{mean(metrics['log_score']):.6f}",
                "log_score_sd": f"{stdev(metrics['log_score']):.6f}",
                "calibration_error_mean": f"{mean(metrics['calibration_error']):.6f}",
                "calibration_error_sd": f"{stdev(metrics['calibration_error']):.6f}",
            }
        )
    with (out_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Aggregate Forecasting Experiment Results",
        "",
        f"Aggregated {len(paths)} run directories.",
        "",
        "| System | Runs | N mean | Brier mean | Brier sd | Log score mean | Calibration error mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['runs']} | {row['n_mean']} | {float(row['brier_mean']):.4f} | "
            f"{float(row['brier_sd']):.4f} | {float(row['log_score_mean']):.4f} | {float(row['calibration_error_mean']):.4f} |"
        )
    lines.extend(["", "## Included Runs", ""])
    for path in paths:
        lines.append(f"- `{path}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecasting_synthetic_aggregate"))
    args = parser.parse_args()
    run_paths = [path for path in args.runs if (path / "metrics.json").exists()]
    if not run_paths:
        raise SystemExit("No run directories with metrics.json found.")
    grouped = load_metrics(run_paths)
    write_summary(grouped, args.out_dir, run_paths)
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()

