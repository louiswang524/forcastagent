from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import AIABaselineForecaster, EvidenceGraphForecaster
from .historical_timeseries_forecastbench import HistoricalAnalogForecaster, HistoricalEvidenceBlendForecaster, build_probabilities
from .metrics import brier_score, log_score, summarize_results
from .models import ForecastResult
from .run_forecasting_experiment import result_to_dict
from .search_llm_forecaster import (
    HeuristicReasoner,
    HybridAnalogSearchForecaster,
    LocalEvidenceSearchProvider,
    OpenAICompatibleReasoner,
    SearchEnabledLLMForecaster,
)


def write_rows(results: list[ForecastResult], outcomes: dict[str, int], out_dir: Path) -> None:
    with (out_dir / "forecast_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "system", "outcome", "probability", "brier", "log_score"])
        writer.writeheader()
        for result in results:
            if result.question_id not in outcomes:
                continue
            probability = result.scored_probability()
            outcome = outcomes[result.question_id]
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "system": result.system,
                    "outcome": outcome,
                    "probability": f"{probability:.6f}",
                    "brier": f"{brier_score(probability, outcome):.6f}",
                    "log_score": f"{log_score(probability, outcome):.6f}",
                }
            )


def write_report(
    out_dir: Path,
    question_set: str,
    targets: int,
    historical_coverage: int,
    reasoner_name: str,
    prompt_style: str,
    temperature: float,
    analog_gate: bool,
    summary: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# ForecastBench Search-Enabled LLM Loop",
        "",
        f"Question set: `{question_set}`",
        f"Targets: `{targets}`",
        f"Historical retrieval coverage: `{historical_coverage}`",
        f"Reasoner: `{reasoner_name}`",
        f"Prompt style: `{prompt_style}`",
        f"Temperature: `{temperature:.2f}`",
        f"Analog gate: `{analog_gate}`",
        "",
        "The search loop plans subqueries, retrieves cutoff-filtered local/historical evidence, runs specialist reasoning agents, and blends their aggregate with the evidence graph.",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in sorted(summary.items(), key=lambda item: item[1]["brier"]):
        lines.append(f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    question_file, resolution_file = download_forecastbench_files(args.raw_dir, args.question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, args.market_subset)
    if args.limit:
        questions = questions[: args.limit]
    historical_probabilities = build_probabilities(questions, args.cache_dir, set(args.sources))
    if args.reasoner == "openai":
        reasoner = OpenAICompatibleReasoner(
            model=args.model,
            base_url=args.openai_base_url,
            api_key_env=args.api_key_env,
            prompt_style=args.prompt_style,
            temperature=args.temperature,
        )
    else:
        reasoner = HeuristicReasoner()
    search_provider = LocalEvidenceSearchProvider(historical_probabilities)
    systems = [
        AIABaselineForecaster(),
        EvidenceGraphForecaster(),
        HistoricalAnalogForecaster(historical_probabilities),
        HistoricalEvidenceBlendForecaster(historical_probabilities, analog_weight=args.analog_weight),
        SearchEnabledLLMForecaster(
            search_provider=search_provider,
            reasoner=reasoner,
            evidence_weight=args.search_weight,
            analog_gate=not args.disable_analog_gate,
        ),
        HybridAnalogSearchForecaster(
            search_provider=search_provider,
            reasoner=reasoner,
            evidence_weight=args.search_weight,
        ),
    ]
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for system in systems:
            results.append(system.forecast(question, evidence))
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    summary = summarize_results(results, outcomes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "historical_probabilities.json").write_text(json.dumps(historical_probabilities, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps([result_to_dict(result) for result in results], indent=2), encoding="utf-8")
    write_rows(results, outcomes, args.out_dir)
    write_report(
        args.out_dir,
        args.question_set,
        len(questions),
        len(historical_probabilities),
        args.reasoner,
        args.prompt_style,
        args.temperature,
        not args.disable_analog_gate,
        summary,
    )
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecastbench_search_llm_20260625"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics", "yfinance"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--analog-weight", type=float, default=1.0)
    parser.add_argument("--search-weight", type=float, default=1.0)
    parser.add_argument("--reasoner", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt-style", choices=sorted(OpenAICompatibleReasoner.SYSTEM_PROMPTS), default="plain")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-analog-gate", action="store_true")
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
