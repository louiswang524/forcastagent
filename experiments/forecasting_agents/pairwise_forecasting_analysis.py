from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def brier(probability: float, outcome: int) -> float:
    return (probability - outcome) ** 2


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def load_rows(paths: list[Path]) -> dict[tuple[str, str], dict[str, tuple[float, int]]]:
    grouped: dict[tuple[str, str], dict[str, tuple[float, int]]] = defaultdict(dict)
    for path in paths:
        run_id = path.name
        with (path / "forecast_rows.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (run_id, row["question_id"])
                grouped[key][row["system"]] = (float(row["probability"]), int(row["outcome"]))
    return grouped


def paired_deltas(
    grouped: dict[tuple[str, str], dict[str, tuple[float, int]]],
    baseline: str,
) -> dict[str, dict[str, float]]:
    deltas: dict[str, list[float]] = defaultdict(list)
    for systems in grouped.values():
        if baseline not in systems:
            continue
        baseline_probability, outcome = systems[baseline]
        baseline_brier = brier(baseline_probability, outcome)
        for system, (probability, system_outcome) in systems.items():
            if system == baseline or system_outcome != outcome:
                continue
            deltas[system].append(brier(probability, outcome) - baseline_brier)
    output: dict[str, dict[str, float]] = {}
    for system, values in deltas.items():
        sd = stdev(values)
        se = sd / math.sqrt(len(values)) if values else 0.0
        output[system] = {
            "n": float(len(values)),
            "mean_delta": mean(values),
            "sd": sd,
            "ci95_low": mean(values) - 1.96 * se,
            "ci95_high": mean(values) + 1.96 * se,
        }
    return output


def write_report(results: dict[str, dict[str, dict[str, float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["baseline", "system", "n", "mean_delta", "sd", "ci95_low", "ci95_high"]
    with (out_dir / "pairwise_brier_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline, rows in results.items():
            for system, row in sorted(rows.items()):
                writer.writerow(
                    {
                        "baseline": baseline,
                        "system": system,
                        "n": int(row["n"]),
                        "mean_delta": f"{row['mean_delta']:.8f}",
                        "sd": f"{row['sd']:.8f}",
                        "ci95_low": f"{row['ci95_low']:.8f}",
                        "ci95_high": f"{row['ci95_high']:.8f}",
                    }
                )
    lines = [
        "# Pairwise Forecasting Analysis",
        "",
        "Negative mean delta means the system has lower Brier score than the baseline on matched questions.",
        "",
    ]
    for baseline, rows in results.items():
        lines.extend([f"## Baseline: `{baseline}`", "", "| System | N | Mean Brier delta | 95% CI |", "|---|---:|---:|---:|"])
        for system, row in sorted(rows.items(), key=lambda item: item[1]["mean_delta"]):
            lines.append(
                f"| {system} | {int(row['n'])} | {row['mean_delta']:.5f} | "
                f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--baselines", nargs="+", default=["aia_baseline", "market_only"])
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecasting_synthetic_pairwise"))
    args = parser.parse_args()
    run_paths = [path for path in args.runs if (path / "forecast_rows.csv").exists()]
    grouped = load_rows(run_paths)
    results = {baseline: paired_deltas(grouped, baseline) for baseline in args.baselines}
    write_report(results, args.out_dir)
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()

