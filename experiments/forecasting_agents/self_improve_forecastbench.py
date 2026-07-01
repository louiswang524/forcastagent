from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import AIABaselineForecaster, CalibratedSourcePriorForecaster, EvidenceGraphForecaster, WeightedEvidenceGraphForecaster
from .metrics import brier_score, summarize_results
from .models import ForecastQuestion, ForecastResult
from .run_forecasting_experiment import result_to_dict


def fit_leave_one_source_priors(questions: list[ForecastQuestion], shrinkage: float = 12.0) -> dict[str, float]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for question in questions:
        source = str(question.metadata.get("forecastbench_source", question.domain))
        if question.outcome is not None:
            by_source[source].append(int(question.outcome))
    all_outcomes = [int(question.outcome) for question in questions if question.outcome is not None]
    global_rate = sum(all_outcomes) / len(all_outcomes)
    priors: dict[str, float] = {}
    for source, outcomes in by_source.items():
        other = [y for other_source, ys in by_source.items() if other_source != source for y in ys]
        pool_rate = sum(other) / len(other) if other else global_rate
        observed = sum(outcomes) / len(outcomes)
        # Conservative blend: most weight stays on out-of-source global rate.
        priors[source] = (sum(outcomes) + shrinkage * pool_rate) / (len(outcomes) + shrinkage)
        priors[f"{source}__observed_rate"] = observed
    priors["__global_rate"] = global_rate
    return priors


def run_candidate_evaluation(args: argparse.Namespace) -> None:
    question_file, resolution_file = download_forecastbench_files(args.raw_dir, args.question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, args.market_subset)
    if args.limit:
        questions = questions[: args.limit]
    priors = fit_leave_one_source_priors(questions, args.shrinkage)
    source_priors = {k: v for k, v in priors.items() if not k.startswith("__") and not k.endswith("__observed_rate")}
    systems = [
        AIABaselineForecaster(),
        EvidenceGraphForecaster(),
        CalibratedSourcePriorForecaster(source_priors),
    ]
    candidate_results = sweep_weighted_graph_candidates(questions, evidence_by_question, source_priors)
    best_candidate = candidate_results[0]["forecaster"]
    systems.append(best_candidate)
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for system in systems:
            results.append(system.forecast(question, evidence))
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    summary = summarize_results(results, outcomes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "source_priors.json").write_text(json.dumps(priors, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps([result_to_dict(r) for r in results], indent=2), encoding="utf-8")
    write_rows(results, outcomes, args.out_dir)
    write_candidate_sweep(candidate_results, args.out_dir)
    write_report(summary, priors, args.out_dir, len(questions), args.question_set, args.market_subset, candidate_results)
    print(args.out_dir / "summary.md")


def sweep_weighted_graph_candidates(
    questions: list[ForecastQuestion],
    evidence_by_question: dict[str, list],
    source_priors: dict[str, float],
) -> list[dict]:
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    configs = []
    for market_weight in [0.0, 0.10, 0.20, 0.35, 0.50]:
        for evidence_multiplier in [0.5, 0.8, 1.0, 1.25]:
            for base_weight in [0.45, 0.60, 0.75]:
                remaining = 1.0 - base_weight
                subweights = {
                    "base_rate": base_weight,
                    "recent_signal": remaining * 0.35,
                    "deadline": remaining * 0.40,
                    "counterevidence": remaining * 0.25,
                }
                configs.append((market_weight, evidence_multiplier, subweights))
    scored = []
    for market_weight, evidence_multiplier, subweights in configs:
        forecaster = WeightedEvidenceGraphForecaster(
            source_priors=source_priors,
            subquestion_weights=subweights,
            market_weight=market_weight,
            evidence_multiplier=evidence_multiplier,
        )
        results = [forecaster.forecast(question, evidence_by_question.get(question.id, [])) for question in questions]
        summary = summarize_results(results, outcomes)[forecaster.name]
        scored.append(
            {
                "forecaster": forecaster,
                "brier": summary["brier"],
                "log_score": summary["log_score"],
                "calibration_error": summary["calibration_error"],
                "market_weight": market_weight,
                "evidence_multiplier": evidence_multiplier,
                "subquestion_weights": subweights,
            }
        )
    scored.sort(key=lambda row: (row["brier"], row["log_score"]))
    return scored


def write_candidate_sweep(candidate_results: list[dict], out_dir: Path) -> None:
    fields = ["rank", "brier", "log_score", "calibration_error", "market_weight", "evidence_multiplier", "base_rate", "recent_signal", "deadline", "counterevidence"]
    with (out_dir / "candidate_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(candidate_results, start=1):
            weights = row["subquestion_weights"]
            writer.writerow(
                {
                    "rank": idx,
                    "brier": f"{row['brier']:.8f}",
                    "log_score": f"{row['log_score']:.8f}",
                    "calibration_error": f"{row['calibration_error']:.8f}",
                    "market_weight": row["market_weight"],
                    "evidence_multiplier": row["evidence_multiplier"],
                    "base_rate": weights["base_rate"],
                    "recent_signal": weights["recent_signal"],
                    "deadline": weights["deadline"],
                    "counterevidence": weights["counterevidence"],
                }
            )


def write_rows(results: list[ForecastResult], outcomes: dict[str, int], out_dir: Path) -> None:
    with (out_dir / "forecast_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "system", "outcome", "probability", "brier"])
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
                }
            )


def write_report(
    summary: dict[str, dict[str, float]],
    priors: dict[str, float],
    out_dir: Path,
    n: int,
    question_set: str,
    market_subset: bool,
    candidate_results: list[dict],
) -> None:
    best = candidate_results[0]
    lines = [
        "# ForecastBench Self-Improvement Candidate",
        "",
        f"Question set: `{question_set}`",
        f"Targets: `{n}`",
        f"Market subset: `{market_subset}`",
        "",
        "Hypothesis H9: source/type priors should be learned from calibration data rather than hand-coded. This run uses leave-one-source-out shrunken priors to avoid directly fitting each source to itself too aggressively.",
        "",
        "Hypothesis H10: the remaining gap is partly aggregation geometry. We sweep source-prior weight, market weight, and evidence multiplier, then report the best candidate on this run.",
        "",
        f"Best weighted candidate: Brier `{best['brier']:.4f}`, market_weight `{best['market_weight']}`, evidence_multiplier `{best['evidence_multiplier']}`, subquestion_weights `{best['subquestion_weights']}`.",
        "",
        "## Metrics",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in sorted(summary.items(), key=lambda item: item[1]["brier"]):
        lines.append(
            f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |"
        )
    lines.extend(["", "## Learned Priors", "", "| Source | Shrunken prior | Observed rate |", "|---|---:|---:|"])
    for source in sorted(k for k in priors if not k.startswith("__") and not k.endswith("__observed_rate")):
        lines.append(f"| {source} | {priors[source]:.4f} | {priors.get(source + '__observed_rate', math.nan):.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-set", default="2024-07-21-human")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecastbench_self_improve_20260625"))
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shrinkage", type=float, default=12.0)
    args = parser.parse_args()
    run_candidate_evaluation(args)


if __name__ == "__main__":
    main()
