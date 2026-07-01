from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .historical_timeseries_forecastbench import build_probabilities
from .metrics import brier_score, calibration_error, log_score, summarize_results
from .models import ForecastQuestion, ForecastResult
from .run_forecasting_experiment import result_to_dict
from .search_llm_forecaster import HeuristicReasoner, HybridAnalogSearchForecaster, LocalEvidenceSearchProvider, SearchEnabledLLMForecaster
from .forecasters import AIABaselineForecaster, EvidenceGraphForecaster, MarketOnlyForecaster
from .historical_timeseries_forecastbench import HistoricalAnalogForecaster, HistoricalEvidenceBlendForecaster


def source_of(question: ForecastQuestion) -> str:
    return str(question.metadata.get("forecastbench_source", question.domain))


def score_rows(results: list[ForecastResult], questions: list[ForecastQuestion]) -> list[dict[str, object]]:
    by_question = {question.id: question for question in questions}
    rows: list[dict[str, object]] = []
    for result in results:
        question = by_question.get(result.question_id)
        if question is None or question.outcome is None:
            continue
        probability = result.scored_probability()
        outcome = int(question.outcome)
        rows.append(
            {
                "question_id": result.question_id,
                "system": result.system,
                "source": source_of(question),
                "horizon_bucket": question.metadata.get("horizon_bucket", "unknown"),
                "outcome": outcome,
                "probability": probability,
                "brier": brier_score(probability, outcome),
                "log_score": log_score(probability, outcome),
                "gate_decision": result.diagnostics.get("gate_decision", "none"),
                "historical_analog_available": result.diagnostics.get("historical_analog_available", result.diagnostics.get("historical_analog_probability") is not None),
            }
        )
    return rows


def summarize_group(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["system"]), str(row[key]))].append(row)
    summary = []
    for (system, value), items in sorted(grouped.items()):
        scored = [(float(item["probability"]), int(item["outcome"])) for item in items]
        summary.append(
            {
                "system": system,
                key: value,
                "n": len(items),
                "brier": sum(float(item["brier"]) for item in items) / len(items),
                "log_score": sum(float(item["log_score"]) for item in items) / len(items),
                "calibration_error": calibration_error(scored),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], key: str, systems: list[str]) -> list[str]:
    filtered = [row for row in rows if row["system"] in systems]
    values = sorted({str(row[key]) for row in filtered})
    by_pair = {(str(row["system"]), str(row[key])): row for row in filtered}
    lines = [f"| {key} | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
    for value in values:
        cells = []
        for system in systems:
            row = by_pair.get((system, value))
            cells.append("" if row is None else f"{float(row['brier']):.4f} ({int(row['n'])})")
        lines.append(f"| {value} | " + " | ".join(cells) + " |")
    return lines


def run(args: argparse.Namespace) -> None:
    question_file, resolution_file = download_forecastbench_files(args.raw_dir, args.question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, args.market_subset)
    probabilities = build_probabilities(questions, args.cache_dir, set(args.sources))
    search_provider = LocalEvidenceSearchProvider(probabilities)
    reasoner = HeuristicReasoner()
    systems = [
        AIABaselineForecaster(),
        MarketOnlyForecaster(),
        EvidenceGraphForecaster(),
        HistoricalAnalogForecaster(probabilities),
        HistoricalEvidenceBlendForecaster(probabilities, analog_weight=1.0),
        SearchEnabledLLMForecaster(search_provider=search_provider, reasoner=reasoner, evidence_weight=1.0, analog_gate=True),
        HybridAnalogSearchForecaster(search_provider=search_provider, reasoner=reasoner, evidence_weight=1.0),
    ]
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for system in systems:
            results.append(system.forecast(question, evidence))
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    overall = summarize_results(results, outcomes)
    rows = score_rows(results, questions)
    source_summary = summarize_group(rows, "source")
    horizon_summary = summarize_group(rows, "horizon_bucket")
    gate_summary = summarize_group([row for row in rows if row["system"] == "hybrid_analog_search"], "gate_decision")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps([result_to_dict(result) for result in results], indent=2), encoding="utf-8")
    write_csv(args.out_dir / "scored_rows.csv", rows)
    write_csv(args.out_dir / "source_summary.csv", source_summary)
    write_csv(args.out_dir / "horizon_summary.csv", horizon_summary)
    write_csv(args.out_dir / "gate_summary.csv", gate_summary)

    systems_for_table = ["aia_baseline", "market_only", "historical_analog", "search_llm_loop", "hybrid_analog_search"]
    lines = [
        "# Conference Gate Experiment",
        "",
        f"Question set: `{args.question_set}`",
        f"Targets: `{len(questions)}`",
        f"Historical coverage: `{len(probabilities)}`",
        f"Sources: `{' '.join(args.sources)}`",
        "",
        "## Overall Metrics",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, metric in sorted(overall.items(), key=lambda item: item[1]["brier"]):
        lines.append(f"| {system} | {int(metric['n'])} | {metric['brier']:.4f} | {metric['log_score']:.4f} | {metric['calibration_error']:.4f} |")
    lines.extend(["", "## Source Brier (N)", ""])
    lines.extend(markdown_table(source_summary, "source", systems_for_table))
    lines.extend(["", "## Hybrid Gate Decision", "", "| Decision | N | Brier | Log score |", "|---|---:|---:|---:|"])
    for row in gate_summary:
        lines.append(f"| {row['gate_decision']} | {int(row['n'])} | {float(row['brier']):.4f} | {float(row['log_score']):.4f} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/conference_gate_20260626"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics", "yfinance"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--market-subset", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
