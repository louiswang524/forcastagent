from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .forecastbench_adapter import (
    download_forecastbench_files,
    download_public_forecaster_file,
    download_superforecaster_file,
    load_forecaster_medians,
    load_forecastbench_targets,
    load_superforecaster_medians,
)
from .forecasters import (
    AIABaselineForecaster,
    AdvancedForecastLab,
    EvidenceGraphForecaster,
    EvidenceGraphNoMarketForecaster,
    MarketOnlyForecaster,
    StructuredEvidenceGraphForecaster,
)
from .metrics import brier_score, calibration_error, log_score, summarize_results
from .models import ForecastResult
from .run_forecasting_experiment import result_to_dict


FORECASTERS = {
    "market_only": MarketOnlyForecaster,
    "aia_baseline": AIABaselineForecaster,
    "evidence_graph_no_market": EvidenceGraphNoMarketForecaster,
    "evidence_graph_v1": EvidenceGraphForecaster,
    "structured_evidence_graph": StructuredEvidenceGraphForecaster,
    "advanced_v1": AdvancedForecastLab,
}


def baseline_result(question_id: str, system: str, probability: float) -> ForecastResult:
    return ForecastResult(
        question_id=question_id,
        system=system,
        probability=probability,
        confidence=0.5,
        rationale=f"Imported ForecastBench baseline: {system}.",
    )


def write_rows(results: list[ForecastResult], outcomes: dict[str, int], out_dir: Path) -> None:
    with (out_dir / "forecast_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "system", "outcome", "probability", "brier", "log_score"])
        writer.writeheader()
        for result in results:
            if result.question_id not in outcomes:
                continue
            p = result.scored_probability()
            y = outcomes[result.question_id]
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "system": result.system,
                    "outcome": y,
                    "probability": f"{p:.6f}",
                    "brier": f"{brier_score(p, y):.6f}",
                    "log_score": f"{log_score(p, y):.6f}",
                }
            )


def write_report(
    summary: dict[str, dict[str, float]],
    out_dir: Path,
    question_set: str,
    targets_count: int,
    market_subset: bool,
    source_counts: dict[str, int],
) -> None:
    lines = [
        "# ForecastBench Experiment",
        "",
        f"Question set: `{question_set}`",
        f"Targets scored: `{targets_count}`",
        f"Market subset only: `{market_subset}`",
        "",
        "This run uses public ForecastBench question/resolution files. It is a structural/offline run of the local prototype systems, not a reproduction of AIA Forecaster's search-enabled LLM forecasts.",
        "",
        "## Source Counts",
        "",
        "| Source | Targets |",
        "|---|---:|",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")
    lines.extend(["", "## Metrics", "", "| System | N | Brier | Log score | Calibration error |", "|---|---:|---:|---:|---:|"])
    for system, row in sorted(summary.items()):
        lines.append(
            f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    question_file, resolution_file = download_forecastbench_files(args.raw_dir, args.question_set)
    questions, evidence_by_question, target_to_base_id = load_forecastbench_targets(question_file, resolution_file, args.market_subset)
    if args.limit:
        questions = questions[: args.limit]
    selected = args.systems or list(FORECASTERS)
    forecasters = [FORECASTERS[name]() for name in selected]
    run_name = args.run_name or time.strftime(f"forecastbench_{args.question_set}_%Y%m%d-%H%M%S")
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for forecaster in forecasters:
            results.append(forecaster.forecast(question, evidence))
    if args.include_superforecasters and args.question_set == "2024-07-21-human":
        super_path = download_superforecaster_file(args.raw_dir)
        medians = load_superforecaster_medians(super_path)
        for question in questions:
            base_id = target_to_base_id.get(question.id, "")
            fallback_key = f"{base_id}::NA"
            if question.id in medians:
                results.append(baseline_result(question.id, "superforecaster_median", medians[question.id]))
            elif fallback_key in medians:
                results.append(baseline_result(question.id, "superforecaster_median", medians[fallback_key]))
    if args.include_public_forecasters and args.question_set == "2024-07-21-human":
        public_path = download_public_forecaster_file(args.raw_dir)
        try:
            medians = load_forecaster_medians(public_path)
        except ValueError as exc:
            print(f"Skipping public_median: {exc}")
        else:
            for question in questions:
                base_id = target_to_base_id.get(question.id, "")
                fallback_key = f"{base_id}::NA"
                if question.id in medians:
                    results.append(baseline_result(question.id, "public_median", medians[question.id]))
                elif fallback_key in medians:
                    results.append(baseline_result(question.id, "public_median", medians[fallback_key]))
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    summary = summarize_results(results, outcomes)
    source_counts: dict[str, int] = {}
    for question in questions:
        source = str(question.metadata.get("forecastbench_source", question.domain))
        source_counts[source] = source_counts.get(source, 0) + 1
    write_rows(results, outcomes, out_dir)
    (out_dir / "results.json").write_text(json.dumps([result_to_dict(result) for result in results], indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir, args.question_set, len(questions), args.market_subset, source_counts)
    print(out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-set", default="2024-07-21-human")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--systems", nargs="+", choices=sorted(FORECASTERS), default=None)
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--include-superforecasters", action="store_true")
    parser.add_argument("--include-public-forecasters", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
