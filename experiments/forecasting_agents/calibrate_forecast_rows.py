from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .metrics import brier_score, calibration_error, log_score
from .models import clamp_probability


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def platt(probability: float, slope: float, intercept: float) -> float:
    return clamp_probability(logistic(slope * logit(probability) + intercept))


def load_rows(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    by_system: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_system[row["system"]].append((row["question_id"], float(row["probability"]), int(row["outcome"])))
    return by_system


def summarize(rows: list[tuple[str, float, int]], slope: float = 1.0, intercept: float = 0.0) -> dict[str, float]:
    scored = [(platt(probability, slope, intercept), outcome) for _, probability, outcome in rows]
    return {
        "n": float(len(scored)),
        "brier": sum(brier_score(probability, outcome) for probability, outcome in scored) / len(scored),
        "log_score": sum(log_score(probability, outcome) for probability, outcome in scored) / len(scored),
        "calibration_error": calibration_error(scored),
    }


def tune_system(rows: list[tuple[str, float, int]], slopes: list[float], intercepts: list[float], objective: str) -> dict[str, float]:
    best: dict[str, float] | None = None
    for slope in slopes:
        for intercept in intercepts:
            metrics = summarize(rows, slope, intercept)
            candidate = {**metrics, "slope": slope, "intercept": intercept}
            if best is None or (candidate[objective], candidate["log_score"]) < (best[objective], best["log_score"]):
                best = candidate
    if best is None:
        raise ValueError("No calibration candidates were evaluated")
    return best


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + step / 10:
        values.append(round(current, 10))
        current += step
    return values


def write_report(
    out_dir: Path,
    train_path: Path,
    eval_path: Path,
    systems: list[str],
    train_best: dict[str, dict[str, float]],
    eval_summary: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Forecast Calibration Sweep",
        "",
        f"Train rows: `{train_path}`",
        f"Eval rows: `{eval_path}`",
        "",
        "Platt/extremization form: `logit(p') = slope * logit(p) + intercept`.",
        "",
        "## Selected Parameters",
        "",
        "| System | Train Brier | Slope | Intercept |",
        "|---|---:|---:|---:|",
    ]
    for system in systems:
        row = train_best[system]
        lines.append(f"| {system} | {row['brier']:.4f} | {row['slope']:.3f} | {row['intercept']:.3f} |")
    lines.extend(
        [
            "",
            "## Evaluation Metrics",
            "",
            "| System | N | Raw Brier | Calibrated Brier | Raw Log | Calibrated Log | Raw Cal error | Calibrated Cal error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system in systems:
        row = eval_summary[system]
        lines.append(
            f"| {system} | {int(row['n'])} | {row['raw_brier']:.4f} | {row['brier']:.4f} | "
            f"{row['raw_log_score']:.4f} | {row['log_score']:.4f} | {row['raw_calibration_error']:.4f} | {row['calibration_error']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    train_rows = load_rows(args.train_rows)
    eval_rows = load_rows(args.eval_rows)
    systems = args.systems or sorted(set(train_rows) & set(eval_rows))
    slopes = frange(args.min_slope, args.max_slope, args.slope_step)
    intercepts = frange(args.min_intercept, args.max_intercept, args.intercept_step)
    train_best: dict[str, dict[str, float]] = {}
    eval_summary: dict[str, dict[str, float]] = {}
    for system in systems:
        if system not in train_rows or system not in eval_rows:
            continue
        best = tune_system(train_rows[system], slopes, intercepts, args.objective)
        raw_eval = summarize(eval_rows[system])
        calibrated_eval = summarize(eval_rows[system], best["slope"], best["intercept"])
        train_best[system] = best
        eval_summary[system] = {
            **calibrated_eval,
            "raw_brier": raw_eval["brier"],
            "raw_log_score": raw_eval["log_score"],
            "raw_calibration_error": raw_eval["calibration_error"],
            "slope": best["slope"],
            "intercept": best["intercept"],
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_best.json").write_text(json.dumps(train_best, indent=2), encoding="utf-8")
    (args.out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    write_report(args.out_dir, args.train_rows, args.eval_rows, list(train_best), train_best, eval_summary)
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--eval-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--objective", choices=["brier", "log_score"], default="brier")
    parser.add_argument("--min-slope", type=float, default=0.55)
    parser.add_argument("--max-slope", type=float, default=2.25)
    parser.add_argument("--slope-step", type=float, default=0.05)
    parser.add_argument("--min-intercept", type=float, default=-1.25)
    parser.add_argument("--max-intercept", type=float, default=1.25)
    parser.add_argument("--intercept-step", type=float, default=0.05)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
