from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import (
    AIABaselineForecaster,
    CalibratedSourcePriorForecaster,
    EvidenceGraphForecaster,
    WeightedEvidenceGraphForecaster,
)
from .metrics import brier_score, summarize_results
from .models import ForecastQuestion, ForecastResult
from .run_forecasting_experiment import result_to_dict
from .self_improve_forecastbench import sweep_weighted_graph_candidates


def fit_hierarchical_priors(questions: list[ForecastQuestion], shrinkage: float = 12.0) -> dict[str, float]:
    by_source: dict[str, list[int]] = defaultdict(list)
    by_source_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for question in questions:
        if question.outcome is None:
            continue
        source = str(question.metadata.get("forecastbench_source", question.domain))
        bucket = str(question.metadata.get("horizon_bucket", "unknown"))
        outcome = int(question.outcome)
        by_source[source].append(outcome)
        by_source_bucket[(source, bucket)].append(outcome)

    all_outcomes = [int(question.outcome) for question in questions if question.outcome is not None]
    global_rate = sum(all_outcomes) / len(all_outcomes)
    priors: dict[str, float] = {"__global_rate": global_rate}
    for source, outcomes in by_source.items():
        source_rate = (sum(outcomes) + shrinkage * global_rate) / (len(outcomes) + shrinkage)
        priors[source] = source_rate
        priors[f"{source}__observed_rate"] = sum(outcomes) / len(outcomes)

    for (source, bucket), outcomes in by_source_bucket.items():
        source_prior = priors[source]
        key = f"{source}::{bucket}"
        priors[key] = (sum(outcomes) + shrinkage * source_prior) / (len(outcomes) + shrinkage)
        priors[f"{key}__observed_rate"] = sum(outcomes) / len(outcomes)
    return priors


def score_systems(
    questions: list[ForecastQuestion],
    evidence_by_question: dict[str, list],
    systems: list,
) -> tuple[list[ForecastResult], dict[str, dict[str, float]]]:
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for system in systems:
            results.append(system.forecast(question, evidence))
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    return results, summarize_results(results, outcomes)


def write_rows(results: list[ForecastResult], outcomes: dict[str, int], out_dir: Path) -> None:
    with (out_dir / "forecast_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "system", "outcome", "probability", "brier"])
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
                }
            )


def write_report(
    out_dir: Path,
    train_set: str,
    eval_set: str,
    train_n: int,
    eval_n: int,
    summary: dict[str, dict[str, float]],
    selected_candidate: dict,
) -> None:
    lines = [
        "# ForecastBench Prior Transfer",
        "",
        f"Calibration set: `{train_set}` (`{train_n}` targets)",
        f"Evaluation set: `{eval_set}` (`{eval_n}` targets)",
        "",
        "This run fits source and source/horizon priors only on the calibration set, selects weighted-graph geometry only on the calibration set, and scores the evaluation set without retuning.",
        "",
        f"Selected weighted config on calibration: Brier `{selected_candidate['brier']:.4f}`, market_weight `{selected_candidate['market_weight']}`, evidence_multiplier `{selected_candidate['evidence_multiplier']}`, subquestion_weights `{selected_candidate['subquestion_weights']}`.",
        "",
        "## Evaluation Metrics",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in sorted(summary.items(), key=lambda item: item[1]["brier"]):
        lines.append(
            f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    train_question_file, train_resolution_file = download_forecastbench_files(args.raw_dir, args.train_question_set)
    eval_question_file, eval_resolution_file = download_forecastbench_files(args.raw_dir, args.eval_question_set)
    train_questions, train_evidence, _ = load_forecastbench_targets(train_question_file, train_resolution_file, args.market_subset)
    eval_questions, eval_evidence, _ = load_forecastbench_targets(eval_question_file, eval_resolution_file, args.market_subset)
    if args.limit:
        train_questions = train_questions[: args.limit]
        eval_questions = eval_questions[: args.limit]

    priors = fit_hierarchical_priors(train_questions, args.shrinkage)
    prior_candidates = {key: value for key, value in priors.items() if not key.startswith("__") and not key.endswith("__observed_rate")}
    selected_candidate = sweep_weighted_graph_candidates(train_questions, train_evidence, prior_candidates)[0]
    weighted = WeightedEvidenceGraphForecaster(
        source_priors=prior_candidates,
        subquestion_weights=selected_candidate["subquestion_weights"],
        market_weight=selected_candidate["market_weight"],
        evidence_multiplier=selected_candidate["evidence_multiplier"],
    )
    systems = [
        AIABaselineForecaster(),
        EvidenceGraphForecaster(),
        CalibratedSourcePriorForecaster({key: value for key, value in prior_candidates.items() if "::" not in key}),
        CalibratedSourcePriorForecaster(prior_candidates),
        weighted,
    ]
    systems[2].name = "transfer_source_prior"
    systems[3].name = "transfer_source_horizon_prior"
    weighted.name = "transfer_weighted_source_horizon"

    results, summary = score_systems(eval_questions, eval_evidence, systems)
    outcomes = {question.id: int(question.outcome) for question in eval_questions if question.outcome is not None}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "source_horizon_priors.json").write_text(json.dumps(priors, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps([result_to_dict(result) for result in results], indent=2), encoding="utf-8")
    write_rows(results, outcomes, args.out_dir)
    write_report(args.out_dir, args.train_question_set, args.eval_question_set, len(train_questions), len(eval_questions), summary, selected_candidate)
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-question-set", default="2024-07-21-human")
    parser.add_argument("--eval-question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecastbench_transfer_20260625"))
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shrinkage", type=float, default=12.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
