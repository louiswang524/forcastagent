from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .metrics import brier_score, calibration_error, log_score
from .models import clamp_probability
from .source_routed_gate_transfer import (
    ReliabilityRoutedForecaster,
    SourceRoutedForecaster,
    TaxonomyRoutedForecaster,
    build_source_features,
    build_systems,
    fit_router,
    fit_reliability_rule,
    load_questions,
    run_systems,
    source_of,
    source_system_metrics,
)


BASE_SYSTEMS = [
    "aia_baseline",
    "market_only",
    "evidence_graph_v1",
    "historical_analog",
    "historical_evidence_blend",
    "search_llm_loop",
    "hybrid_analog_search",
]


def logit(probability: float) -> float:
    probability = clamp_probability(probability)
    return math.log(probability / (1.0 - probability))


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def calibrate(probability: float, slope: float, intercept: float) -> float:
    return clamp_probability(logistic(slope * logit(probability) + intercept))


def result_rows(results, questions) -> list[dict[str, object]]:
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
                "horizon_bucket": str(question.metadata.get("horizon_bucket", "unknown")),
                "outcome": outcome,
                "probability": probability,
                "brier": brier_score(probability, outcome),
                "log_score": log_score(probability, outcome),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]], system: str) -> dict[str, float]:
    items = [row for row in rows if row["system"] == system]
    scored = [(float(row["probability"]), int(row["outcome"])) for row in items]
    return {
        "n": float(len(items)),
        "brier": sum(float(row["brier"]) for row in items) / len(items),
        "log_score": sum(float(row["log_score"]) for row in items) / len(items),
        "calibration_error": calibration_error(scored),
    }


def add_calibrated_rows(
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    source_min_n: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    train_taxonomy = [row for row in train_rows if row["system"] == "taxonomy_routed_gate"]
    global_params = fit_calibration(train_taxonomy)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train_taxonomy:
        by_source[str(row["source"])].append(row)
    source_params = {
        source: fit_calibration(rows) if len(rows) >= source_min_n else global_params
        for source, rows in by_source.items()
    }

    added: list[dict[str, object]] = []
    for row in eval_rows:
        if row["system"] != "taxonomy_routed_gate":
            continue
        for system, params in [
            ("taxonomy_global_calibrated", global_params),
            ("taxonomy_source_calibrated", source_params.get(str(row["source"]), global_params)),
        ]:
            probability = calibrate(float(row["probability"]), params["slope"], params["intercept"])
            outcome = int(row["outcome"])
            added.append(
                {
                    **row,
                    "system": system,
                    "probability": probability,
                    "brier": brier_score(probability, outcome),
                    "log_score": log_score(probability, outcome),
                }
            )
    params = {"global": global_params, **{f"source::{k}": v for k, v in sorted(source_params.items())}}
    return added, params


def fit_calibration(rows: list[dict[str, object]]) -> dict[str, float]:
    slopes = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
    intercepts = [x / 10.0 for x in range(-8, 9)]
    best = {"slope": 1.0, "intercept": 0.0, "train_log_score": float("inf")}
    for slope in slopes:
        for intercept in intercepts:
            losses = [
                log_score(calibrate(float(row["probability"]), slope, intercept), int(row["outcome"]))
                for row in rows
            ]
            score = sum(losses) / len(losses)
            if score < best["train_log_score"]:
                best = {"slope": slope, "intercept": intercept, "train_log_score": score}
    return best


def add_oracle_rows(eval_rows: list[dict[str, object]], allowed_systems: list[str]) -> list[dict[str, object]]:
    by_question: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        if row["system"] in allowed_systems:
            by_question[str(row["question_id"])].append(row)
            by_source[str(row["source"])].append(row)

    source_choice = {}
    for source, rows in by_source.items():
        by_system: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_system[str(row["system"])].append(row)
        source_choice[source] = min(
            by_system,
            key=lambda system: sum(float(row["brier"]) for row in by_system[system]) / len(by_system[system]),
        )

    added: list[dict[str, object]] = []
    for question_id, rows in by_question.items():
        best_question = min(rows, key=lambda row: float(row["brier"]))
        source_selected = next(row for row in rows if row["system"] == source_choice[str(row["source"])])
        for system, selected in [
            ("oracle_question_router", best_question),
            ("oracle_source_router", source_selected),
        ]:
            added.append({**selected, "system": system})
    return added


def paired_bootstrap(
    rows: list[dict[str, object]],
    reference: str,
    comparators: list[str],
    metric: str,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    by_question: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_question[str(row["question_id"])][str(row["system"])] = float(row[metric])
    question_ids = [
        question_id
        for question_id, values in by_question.items()
        if reference in values and all(comparator in values for comparator in comparators)
    ]
    output = []
    for comparator in comparators:
        observed = sum(by_question[q][reference] - by_question[q][comparator] for q in question_ids) / len(question_ids)
        diffs = []
        for _ in range(samples):
            draw = [rng.choice(question_ids) for _ in question_ids]
            diffs.append(sum(by_question[q][reference] - by_question[q][comparator] for q in draw) / len(draw))
        diffs.sort()
        p_reference_worse_or_equal = sum(1 for value in diffs if value >= 0.0) / len(diffs)
        output.append(
            {
                "reference": reference,
                "comparator": comparator,
                "metric": metric,
                "n": len(question_ids),
                "mean_diff": observed,
                "ci_low": diffs[int(0.025 * len(diffs))],
                "ci_high": diffs[int(0.975 * len(diffs))],
                "p_reference_worse_or_equal": p_reference_worse_or_equal,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    metrics: dict[str, dict[str, float]],
    bootstrap_rows: list[dict[str, object]],
    calibration_params: dict[str, dict[str, float]],
) -> None:
    ordered = sorted(metrics.items(), key=lambda item: item[1]["brier"])
    lines = [
        "# Paper Quality Analysis",
        "",
        "## Routing and Calibration Ablations",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in ordered:
        lines.append(
            f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Bootstrap Differences",
            "",
            "Differences are `reliability_routed_gate - comparator`; negative values favor the reliability router.",
            "",
            "| Comparator | Metric | Mean diff | 95% CI | P(diff >= 0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['comparator']} | {row['metric']} | {row['mean_diff']:.5f} | [{row['ci_low']:.5f}, {row['ci_high']:.5f}] | {row['p_reference_worse_or_equal']:.3f} |"
        )
    lines.extend(["", "## Calibration Parameters", "", "| Scope | Slope | Intercept | Train log score |", "|---|---:|---:|---:|"])
    for scope, params in calibration_params.items():
        lines.append(f"| {scope} | {params['slope']:.2f} | {params['intercept']:.2f} | {params['train_log_score']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    train_questions, train_evidence = load_questions(args.raw_dir, args.train_question_set, args.market_subset)
    eval_questions, eval_evidence = load_questions(args.raw_dir, args.eval_question_set, args.market_subset)

    train_systems, train_probabilities = build_systems(train_questions, args.cache_dir, args.sources)
    train_source_features = build_source_features(train_questions, train_evidence, train_probabilities)
    train_results = run_systems(train_systems, train_questions, train_evidence)
    train_source_rows = source_system_metrics(train_results, train_questions)
    source_map, default_system = fit_router(
        train_source_rows,
        ["aia_baseline", "market_only", "evidence_graph_v1", "historical_analog", "search_llm_loop"],
        args.min_source_n,
    )
    reliability_rule = fit_reliability_rule(
        train_results,
        train_questions,
        train_evidence,
        train_probabilities,
        train_source_features,
        {"aia_baseline", "market_only", "evidence_graph_v1", "historical_analog", "search_llm_loop"},
    )
    train_systems["taxonomy_routed_gate"] = TaxonomyRoutedForecaster(train_systems)
    train_systems["reliability_routed_gate"] = ReliabilityRoutedForecaster(train_systems, train_probabilities, train_source_features, reliability_rule)
    train_results = run_systems(train_systems, train_questions, train_evidence)

    eval_systems, eval_probabilities = build_systems(eval_questions, args.cache_dir, args.sources)
    eval_source_features = build_source_features(eval_questions, eval_evidence, eval_probabilities)
    eval_systems["source_routed_gate"] = SourceRoutedForecaster(eval_systems, source_map, default_system)
    eval_systems["taxonomy_routed_gate"] = TaxonomyRoutedForecaster(eval_systems)
    eval_systems["reliability_routed_gate"] = ReliabilityRoutedForecaster(eval_systems, eval_probabilities, eval_source_features, reliability_rule)
    eval_results = run_systems(eval_systems, eval_questions, eval_evidence)

    train_rows = result_rows(train_results, train_questions)
    eval_rows = result_rows(eval_results, eval_questions)
    calibrated_rows, calibration_params = add_calibrated_rows(train_rows, eval_rows, args.source_calibration_min_n)
    oracle_rows = add_oracle_rows(eval_rows, BASE_SYSTEMS)
    all_rows = eval_rows + calibrated_rows + oracle_rows

    systems = sorted({str(row["system"]) for row in all_rows})
    metrics = {system: summarize(all_rows, system) for system in systems}
    comparators = [
        "taxonomy_routed_gate",
        "historical_analog",
        "search_llm_loop",
        "source_routed_gate",
        "aia_baseline",
        "taxonomy_global_calibrated",
        "taxonomy_source_calibrated",
    ]
    bootstrap_rows = []
    for metric in ["brier", "log_score"]:
        bootstrap_rows.extend(paired_bootstrap(all_rows, "reliability_routed_gate", comparators, metric, args.bootstrap_samples, args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out_dir / "bootstrap.json").write_text(json.dumps(bootstrap_rows, indent=2), encoding="utf-8")
    (args.out_dir / "calibration_params.json").write_text(json.dumps(calibration_params, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "eval_rows_augmented.csv", all_rows)
    write_csv(args.out_dir / "bootstrap.csv", bootstrap_rows)
    write_markdown(args.out_dir / "summary.md", metrics, bootstrap_rows, calibration_params)
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-question-set", default="2024-07-21-human")
    parser.add_argument("--eval-question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/paper_quality_20260626"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics", "yfinance"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--min-source-n", type=int, default=8)
    parser.add_argument("--source-calibration-min-n", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
