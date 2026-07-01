from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from .metrics import brier_score, calibration_error, log_score, summarize_results
from .run_forecasting_experiment import result_to_dict
from .source_routed_gate_transfer import (
    ReliabilityRoutedForecaster,
    ReliabilityRule,
    SourceRoutedForecaster,
    TaxonomyRoutedForecaster,
    build_source_features,
    build_systems,
    fit_reliability_rule,
    fit_router,
    load_questions,
    run_systems,
    source_of,
    source_system_metrics,
)


BASE_SYSTEMS = ["aia_baseline", "market_only", "evidence_graph_v1", "historical_analog", "search_llm_loop"]


def clipped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def perturb_rules(rule: ReliabilityRule) -> list[tuple[str, ReliabilityRule]]:
    variants: list[tuple[str, ReliabilityRule]] = [("base", rule)]
    variants.extend(
        [
            ("hist_coverage_minus_0.10", replace(rule, min_hist_coverage=clipped(rule.min_hist_coverage - 0.10, 0.0, 1.0))),
            ("hist_coverage_plus_0.10", replace(rule, min_hist_coverage=clipped(rule.min_hist_coverage + 0.10, 0.0, 1.0))),
            ("market_sharpness_minus_0.05", replace(rule, market_override_sharpness=max(0.0, rule.market_override_sharpness - 0.05))),
            ("market_sharpness_plus_0.05", replace(rule, market_override_sharpness=rule.market_override_sharpness + 0.05)),
            ("prior_sharpness_minus_0.06", replace(rule, prior_override_sharpness=max(0.0, rule.prior_override_sharpness - 0.06))),
            ("prior_sharpness_plus_0.06", replace(rule, prior_override_sharpness=rule.prior_override_sharpness + 0.06)),
            ("source_prior_minus_0.06", replace(rule, source_prior_route_sharpness=max(0.0, rule.source_prior_route_sharpness - 0.06))),
            ("source_prior_plus_0.06", replace(rule, source_prior_route_sharpness=rule.source_prior_route_sharpness + 0.06)),
            ("noisy_strength_minus_0.05", replace(rule, noisy_source_strength=max(0.0, rule.noisy_source_strength - 0.05))),
            ("noisy_strength_plus_0.05", replace(rule, noisy_source_strength=rule.noisy_source_strength + 0.05)),
            ("horizon_150", replace(rule, max_hist_horizon_days=150)),
            ("horizon_9999", replace(rule, max_hist_horizon_days=9999)),
            ("search_enabled_low", replace(rule, search_strength=0.30)),
        ]
    )
    dedup: dict[str, ReliabilityRule] = {}
    for name, variant in variants:
        dedup.setdefault(name, variant)
    return list(dedup.items())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def route_rows(results, questions, question_set: str, router_name: str) -> list[dict[str, object]]:
    by_question = {question.id: question for question in questions}
    rows: list[dict[str, object]] = []
    for result in results:
        if result.system != router_name:
            continue
        question = by_question.get(result.question_id)
        if question is None:
            continue
        selected = str(result.diagnostics.get("selected_system", "unknown"))
        rows.append(
            {
                "question_set": question_set,
                "question_id": result.question_id,
                "source": source_of(question),
                "selected_system": selected,
                "outcome": int(question.outcome),
                "probability": result.scored_probability(),
                "brier": brier_score(result.scored_probability(), int(question.outcome)),
                "hist_available": result.diagnostics.get("hist_available", ""),
                "market_available": result.diagnostics.get("market_available", ""),
                "market_sharpness": result.diagnostics.get("market_sharpness", ""),
                "base_prior_sharpness": result.diagnostics.get("base_prior_sharpness", ""),
                "horizon_days": result.diagnostics.get("horizon_days", ""),
                "evidence_strength": result.diagnostics.get("evidence_strength", ""),
                "evidence_disagreement": result.diagnostics.get("evidence_disagreement", ""),
                "source_hist_coverage": result.diagnostics.get("source_hist_coverage", ""),
                "source_prior_sharpness": result.diagnostics.get("source_prior_sharpness", ""),
            }
        )
    return rows


def summarize_route_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], int] = Counter()
    for row in rows:
        grouped[(str(row["question_set"]), str(row["source"]), str(row["selected_system"]))] += 1
    return [
        {"question_set": question_set, "source": source, "selected_system": system, "n": n}
        for (question_set, source, system), n in sorted(grouped.items())
    ]


def summarize_behavior(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["question_set"]), str(row["selected_system"]))].append(row)
    output = []
    for (question_set, selected), items in sorted(grouped.items()):
        scored = [(float(item["probability"]), int(item["outcome"])) for item in items]
        output.append(
            {
                "question_set": question_set,
                "selected_system": selected,
                "n": len(items),
                "mean_brier": sum(float(item["brier"]) for item in items) / len(items),
                "calibration_error": calibration_error(scored),
                "mean_market_sharpness": mean_feature(items, "market_sharpness"),
                "mean_base_prior_sharpness": mean_feature(items, "base_prior_sharpness"),
                "mean_horizon_days": mean_feature(items, "horizon_days"),
                "mean_evidence_strength": mean_feature(items, "evidence_strength"),
                "mean_source_hist_coverage": mean_feature(items, "source_hist_coverage"),
            }
        )
    return output


def mean_feature(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] != ""]
    return sum(values) / max(1, len(values))


def run(args: argparse.Namespace) -> None:
    train_questions, train_evidence = load_questions(args.raw_dir, args.train_question_set, args.market_subset)
    train_systems, train_probabilities = build_systems(train_questions, args.cache_dir, args.sources)
    train_source_features = build_source_features(train_questions, train_evidence, train_probabilities)
    train_results = run_systems(train_systems, train_questions, train_evidence)
    train_source_rows = source_system_metrics(train_results, train_questions)
    source_map, default_system = fit_router(train_source_rows, BASE_SYSTEMS, args.min_source_n)
    reliability_rule = fit_reliability_rule(
        train_results,
        train_questions,
        train_evidence,
        train_probabilities,
        train_source_features,
        set(BASE_SYSTEMS),
        self_adjust_thresholds=args.self_adjust_thresholds,
    )

    metric_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    all_route_rows: list[dict[str, object]] = []
    all_route_count_rows: list[dict[str, object]] = []
    all_behavior_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []

    calibration_questions = list(train_questions)
    calibration_evidence = dict(train_evidence)
    calibration_probabilities = dict(train_probabilities)
    calibration_results = [result for result in train_results if result.system in BASE_SYSTEMS]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for question_set in args.eval_question_sets:
        if args.self_adjust_thresholds and args.walk_forward_adjustment:
            calibration_source_features = build_source_features(calibration_questions, calibration_evidence, calibration_probabilities)
            reliability_rule = fit_reliability_rule(
                calibration_results,
                calibration_questions,
                calibration_evidence,
                calibration_probabilities,
                calibration_source_features,
                set(BASE_SYSTEMS),
                self_adjust_thresholds=True,
            )
        rule_rows.append({"question_set": question_set, **reliability_rule.__dict__})
        rule_variants = perturb_rules(reliability_rule)

        questions, evidence = load_questions(args.raw_dir, question_set, args.market_subset)
        systems, probabilities = build_systems(questions, args.cache_dir, args.sources)
        source_features = build_source_features(questions, evidence, probabilities)
        systems["source_routed_gate"] = SourceRoutedForecaster(systems, source_map, default_system)
        systems["taxonomy_routed_gate"] = TaxonomyRoutedForecaster(systems)
        systems["reliability_routed_gate"] = ReliabilityRoutedForecaster(systems, probabilities, source_features, reliability_rule)
        results = run_systems(systems, questions, evidence)
        outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
        summary = summarize_results(results, outcomes)
        for system, values in sorted(summary.items()):
            metric_rows.append({"question_set": question_set, "system": system, **values})

        route_trace = route_rows(results, questions, question_set, "reliability_routed_gate")
        all_route_rows.extend(route_trace)
        all_route_count_rows.extend(summarize_route_counts(route_trace))
        all_behavior_rows.extend(summarize_behavior(route_trace))
        (args.out_dir / f"{question_set}_eval_results.json").write_text(
            json.dumps([result_to_dict(result) for result in results], indent=2),
            encoding="utf-8",
        )

        base_choices = {row["question_id"]: row["selected_system"] for row in route_trace}
        base_summary = summary["reliability_routed_gate"]
        for variant_name, variant_rule in rule_variants:
            variant_systems = dict(systems)
            variant_systems["reliability_variant"] = ReliabilityRoutedForecaster(variant_systems, probabilities, source_features, variant_rule)
            variant_results = [
                variant_systems["reliability_variant"].forecast(question, evidence.get(question.id, []))
                for question in questions
            ]
            variant_summary = summarize_results(variant_results, outcomes)["reliability_routed_gate"]
            variant_choices = {
                result.question_id: str(result.diagnostics.get("selected_system", "unknown"))
                for result in variant_results
            }
            common_ids = sorted(set(base_choices) & set(variant_choices))
            changed = sum(1 for question_id in common_ids if base_choices[question_id] != variant_choices[question_id])
            stability_rows.append(
                {
                    "question_set": question_set,
                    "variant": variant_name,
                    "n": len(common_ids),
                    "route_change_rate": changed / max(1, len(common_ids)),
                    "brier": variant_summary["brier"],
                    "delta_brier_vs_base": variant_summary["brier"] - base_summary["brier"],
                    "log_score": variant_summary["log_score"],
                    "delta_log_vs_base": variant_summary["log_score"] - base_summary["log_score"],
                }
            )

        if args.self_adjust_thresholds and args.walk_forward_adjustment:
            calibration_questions.extend(questions)
            calibration_evidence.update(evidence)
            calibration_probabilities.update(probabilities)
            calibration_results.extend(result for result in results if result.system in BASE_SYSTEMS)

    write_csv(args.out_dir / "multi_vintage_metrics.csv", metric_rows)
    write_csv(args.out_dir / "route_traces.csv", all_route_rows)
    write_csv(args.out_dir / "route_counts.csv", all_route_count_rows)
    write_csv(args.out_dir / "behavior_trace_summary.csv", all_behavior_rows)
    write_csv(args.out_dir / "threshold_stability.csv", stability_rows)
    write_csv(args.out_dir / "reliability_rules_by_vintage.csv", rule_rows)
    (args.out_dir / "reliability_rule.json").write_text(json.dumps(reliability_rule.__dict__, indent=2), encoding="utf-8")
    write_report(
        args.out_dir,
        args.train_question_set,
        args.eval_question_sets,
        metric_rows,
        stability_rows,
        all_behavior_rows,
        reliability_rule,
        rule_rows,
        args.self_adjust_thresholds,
        args.walk_forward_adjustment,
    )
    print(args.out_dir / "summary.md")


def write_report(
    out_dir: Path,
    train_set: str,
    eval_sets: list[str],
    metric_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    behavior_rows: list[dict[str, object]],
    reliability_rule: ReliabilityRule,
    rule_rows: list[dict[str, object]],
    self_adjust_thresholds: bool,
    walk_forward_adjustment: bool,
) -> None:
    systems = [
        "taxonomy_routed_gate",
        "reliability_routed_gate",
        "historical_analog",
        "search_llm_loop",
        "source_routed_gate",
        "aia_baseline",
        "evidence_graph_v1",
    ]
    by_pair = {(str(row["question_set"]), str(row["system"])): row for row in metric_rows}
    lines = [
        "# Multi-Vintage Behavioral Routing Analysis",
        "",
        f"Calibration set: `{train_set}`",
        f"Evaluation sets: {', '.join(f'`{item}`' for item in eval_sets)}",
        f"Self-adjusting thresholds: `{self_adjust_thresholds}`",
        f"Walk-forward adjustment: `{walk_forward_adjustment}`",
        "",
        "## Final Reliability Rule",
        "",
        "```json",
        json.dumps(reliability_rule.__dict__, indent=2),
        "```",
        "",
        "## Multi-Vintage Metrics",
        "",
        "| Question set | " + " | ".join(f"{system} Brier" for system in systems) + " |",
        "|---|" + "---:|" * len(systems),
    ]
    for question_set in eval_sets:
        cells = []
        for system in systems:
            row = by_pair.get((question_set, system))
            cells.append("" if row is None else f"{float(row['brier']):.4f}")
        lines.append(f"| {question_set} | " + " | ".join(cells) + " |")

    base_stability = [row for row in stability_rows if row["variant"] != "base"]
    if base_stability:
        lines.extend(
            [
                "",
                "## Threshold Stability",
                "",
                "| Question set | Mean route-change rate | Worst route-change rate | Mean abs delta Brier | Worst abs delta Brier |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        by_set: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in base_stability:
            by_set[str(row["question_set"])].append(row)
        for question_set in eval_sets:
            rows = by_set.get(question_set, [])
            if not rows:
                continue
            route_changes = [float(row["route_change_rate"]) for row in rows]
            delta_briers = [abs(float(row["delta_brier_vs_base"])) for row in rows]
            lines.append(
                f"| {question_set} | {sum(route_changes)/len(route_changes):.3f} | {max(route_changes):.3f} | "
                f"{sum(delta_briers)/len(delta_briers):.4f} | {max(delta_briers):.4f} |"
            )

    if rule_rows:
        lines.extend(
            [
                "",
                "## Rule Evolution",
                "",
                "| Question set | Hist coverage | Market sharpness | Prior sharpness | Source-prior sharpness | Noisy strength | Horizon | Search strength | Search disagreement |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rule_rows:
            lines.append(
                f"| {row['question_set']} | {float(row['min_hist_coverage']):.3f} | {float(row['market_override_sharpness']):.3f} | "
                f"{float(row['prior_override_sharpness']):.3f} | {float(row['source_prior_route_sharpness']):.3f} | "
                f"{float(row['noisy_source_strength']):.3f} | {int(row['max_hist_horizon_days'])} | "
                f"{float(row['search_strength']):.3f} | {float(row['max_search_disagreement']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Behavioral Trace Summary",
            "",
            "| Question set | Selected behavior | N | Mean Brier | Mean horizon | Mean source hist. coverage | Mean evidence strength |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in behavior_rows:
        lines.append(
            f"| {row['question_set']} | {row['selected_system']} | {int(row['n'])} | {float(row['mean_brier']):.4f} | "
            f"{float(row['mean_horizon_days']):.1f} | {float(row['mean_source_hist_coverage']):.3f} | {float(row['mean_evidence_strength']):.3f} |"
        )
    out_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-question-set", default="2024-07-21-human")
    parser.add_argument(
        "--eval-question-sets",
        nargs="+",
        default=[
            "2025-03-02-llm",
            "2025-03-16-llm",
            "2025-03-30-llm",
            "2025-04-13-llm",
            "2025-04-27-llm",
            "2025-05-11-llm",
            "2025-05-25-llm",
            "2025-06-08-llm",
            "2025-06-22-llm",
            "2025-08-03-llm",
            "2025-08-17-llm",
            "2025-08-31-llm",
            "2025-10-26-llm",
            "2025-11-09-llm",
            "2025-11-23-llm",
            "2025-12-07-llm",
        ],
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/multivintage_behavior_20260627"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics", "yfinance"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--min-source-n", type=int, default=8)
    parser.add_argument(
        "--self-adjust-thresholds",
        action="store_true",
        help="Add calibration-distribution quantiles to the ReliabilityRoute threshold grid.",
    )
    parser.add_argument(
        "--walk-forward-adjustment",
        action="store_true",
        help="Before each evaluation vintage, refit ReliabilityRoute thresholds using only the original calibration set and earlier evaluation vintages.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
