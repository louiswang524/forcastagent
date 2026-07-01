from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import AIABaselineForecaster, EvidenceGraphForecaster, MarketOnlyForecaster
from .historical_timeseries_forecastbench import HistoricalAnalogForecaster, HistoricalEvidenceBlendForecaster, build_probabilities
from .metrics import brier_score, calibration_error, log_score, summarize_results
from .models import EvidenceItem, ForecastQuestion, ForecastResult
from .run_forecasting_experiment import result_to_dict
from .search_llm_forecaster import HeuristicReasoner, HybridAnalogSearchForecaster, LocalEvidenceSearchProvider, SearchEnabledLLMForecaster


class ForecastingSystem(Protocol):
    name: str

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        ...


class SourceRoutedForecaster:
    name = "source_routed_gate"

    def __init__(self, systems: dict[str, ForecastingSystem], source_to_system: dict[str, str], default_system: str) -> None:
        self.systems = systems
        self.source_to_system = source_to_system
        self.default_system = default_system

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        source = source_of(question)
        selected_name = self.source_to_system.get(source, self.default_system)
        selected = self.systems[selected_name]
        result = selected.forecast(question, evidence)
        return ForecastResult(
            question_id=result.question_id,
            system=self.name,
            probability=result.probability,
            confidence=result.confidence,
            rationale=f"Source-routed gate selected {selected_name} for ForecastBench source {source}.",
            component_forecasts=result.component_forecasts,
            evidence=result.evidence,
            subquestions=result.subquestions,
            diagnostics={**result.diagnostics, "routed_source": source, "selected_system": selected_name},
        )


class TaxonomyRoutedForecaster:
    name = "taxonomy_routed_gate"

    def __init__(self, systems: dict[str, ForecastingSystem]) -> None:
        self.systems = systems

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        source = source_of(question)
        if source in {"dbnomics", "yfinance", "infer", "manifold", "metaculus", "polymarket"}:
            selected_name = "aia_baseline"
        else:
            selected_name = "historical_analog"
        selected = self.systems[selected_name]
        result = selected.forecast(question, evidence)
        return ForecastResult(
            question_id=result.question_id,
            system=self.name,
            probability=result.probability,
            confidence=result.confidence,
            rationale=f"Taxonomy gate selected {selected_name} for source family {source}.",
            component_forecasts=result.component_forecasts,
            evidence=result.evidence,
            subquestions=result.subquestions,
            diagnostics={**result.diagnostics, "routed_source": source, "selected_system": selected_name},
        )


@dataclass(frozen=True)
class ReliabilityRule:
    min_hist_coverage: float
    market_override_sharpness: float
    prior_override_sharpness: float
    source_prior_route_sharpness: float
    noisy_source_strength: float
    max_hist_horizon_days: int
    search_strength: float
    max_search_disagreement: float
    fallback_system: str = "aia_baseline"


class ReliabilityRoutedForecaster:
    name = "reliability_routed_gate"

    def __init__(
        self,
        systems: dict[str, ForecastingSystem],
        historical_probabilities: dict[str, float],
        source_features: dict[str, dict[str, float]],
        rule: ReliabilityRule,
    ) -> None:
        self.systems = systems
        self.historical_probabilities = historical_probabilities
        self.source_features = source_features
        self.rule = rule

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        features = reliability_features(question, evidence, self.historical_probabilities, self.source_features)
        selected_name = route_from_features(features, self.rule)
        result = self.systems[selected_name].forecast(question, evidence)
        return ForecastResult(
            question_id=result.question_id,
            system=self.name,
            probability=result.probability,
            confidence=result.confidence,
            rationale=f"Reliability router selected {selected_name} from non-label features.",
            component_forecasts=result.component_forecasts,
            evidence=result.evidence,
            subquestions=result.subquestions,
            diagnostics={**result.diagnostics, "selected_system": selected_name, **features},
        )


def source_of(question: ForecastQuestion) -> str:
    return str(question.metadata.get("forecastbench_source", question.domain))


def horizon_days(question: ForecastQuestion) -> int:
    try:
        start = datetime.fromisoformat(str(question.metadata.get("forecast_due_date", question.cutoff_date))[:10])
        end = datetime.fromisoformat(question.resolution_date[:10])
    except ValueError:
        return 9999
    return max(1, (end - start).days)


def evidence_stats(evidence: list[EvidenceItem]) -> tuple[float, float]:
    support = sum(item.weight for item in evidence if item.stance == "supports")
    oppose = sum(item.weight for item in evidence if item.stance == "opposes")
    total = support + oppose
    disagreement = 0.0 if total <= 0 else min(support, oppose) / total
    return total, disagreement


def build_source_features(
    questions: list[ForecastQuestion],
    evidence_by_question: dict[str, list[EvidenceItem]],
    historical_probabilities: dict[str, float],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ForecastQuestion]] = defaultdict(list)
    for question in questions:
        grouped[source_of(question)].append(question)
    output: dict[str, dict[str, float]] = {}
    for source, items in grouped.items():
        n = len(items)
        output[source] = {
            "source_n": float(n),
            "source_hist_coverage": sum(1 for question in items if question.id in historical_probabilities) / max(1, n),
            "source_market_rate": sum(1 for question in items if question.market_probability is not None) / max(1, n),
            "source_prior_sharpness": sum(abs(float(question.metadata.get("base_prior", 0.5)) - 0.5) for question in items) / max(1, n),
            "source_avg_horizon_days": sum(horizon_days(question) for question in items) / max(1, n),
            "source_avg_evidence_strength": sum(evidence_stats(evidence_by_question.get(question.id, []))[0] for question in items) / max(1, n),
        }
    return output


def reliability_features(
    question: ForecastQuestion,
    evidence: list[EvidenceItem],
    historical_probabilities: dict[str, float],
    source_features: dict[str, dict[str, float]],
) -> dict[str, float]:
    source = source_of(question)
    source_row = source_features.get(source, {})
    evidence_strength, disagreement = evidence_stats(evidence)
    market_probability = question.market_probability
    return {
        "hist_available": 1.0 if question.id in historical_probabilities else 0.0,
        "market_available": 1.0 if market_probability is not None else 0.0,
        "market_sharpness": 0.0 if market_probability is None else abs(market_probability - 0.5),
        "base_prior_sharpness": abs(float(question.metadata.get("base_prior", 0.5)) - 0.5),
        "horizon_days": float(horizon_days(question)),
        "evidence_strength": evidence_strength,
        "evidence_disagreement": disagreement,
        "source_hist_coverage": float(source_row.get("source_hist_coverage", 0.0)),
        "source_market_rate": float(source_row.get("source_market_rate", 0.0)),
        "source_prior_sharpness": float(source_row.get("source_prior_sharpness", 0.0)),
        "source_avg_horizon_days": float(source_row.get("source_avg_horizon_days", 9999.0)),
        "source_avg_evidence_strength": float(source_row.get("source_avg_evidence_strength", 0.0)),
    }


def route_from_features(features: dict[str, float], rule: ReliabilityRule) -> str:
    has_market = features["market_available"] > 0.5
    strong_market = has_market and features["market_sharpness"] >= rule.market_override_sharpness
    sharp_prior = (
        features["base_prior_sharpness"] >= rule.prior_override_sharpness
        or features["source_prior_sharpness"] >= rule.source_prior_route_sharpness
    )
    reliable_history = (
        features["hist_available"] > 0.5
        and features["source_hist_coverage"] >= rule.min_hist_coverage
        and features["horizon_days"] <= rule.max_hist_horizon_days
    )
    useful_search = (
        features["evidence_strength"] >= rule.search_strength
        and features["evidence_disagreement"] <= rule.max_search_disagreement
    )
    if strong_market:
        return "aia_baseline"
    if reliable_history and features["source_avg_evidence_strength"] >= rule.noisy_source_strength:
        return "aia_baseline"
    if reliable_history:
        return "historical_analog"
    if sharp_prior:
        return "historical_analog"
    if useful_search and not has_market:
        return "search_llm_loop"
    return rule.fallback_system


def _dedup_values(values: list[float], precision: int = 6) -> list[float]:
    seen = set()
    output = []
    for value in values:
        key = round(float(value), precision)
        if key in seen:
            continue
        seen.add(key)
        output.append(float(value))
    return sorted(output)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def _feature_thresholds(
    feature_rows: list[dict[str, float]],
    key: str,
    base: list[float],
    qs: list[float],
    max_values: int | None = None,
) -> list[float]:
    values = [float(row[key]) for row in feature_rows if key in row]
    if not values:
        return base
    adaptive = [_quantile(values, q) for q in qs]
    output = _dedup_values(base + adaptive)
    if max_values is None or len(output) <= max_values:
        return output
    keep = list(base[: max(1, min(len(base), max_values))])
    for value in adaptive:
        if len(_dedup_values(keep)) >= max_values:
            break
        keep.append(value)
    return _dedup_values(keep)[:max_values]


def reliability_feature_rows(
    questions: list[ForecastQuestion],
    evidence_by_question: dict[str, list[EvidenceItem]],
    historical_probabilities: dict[str, float],
    source_features: dict[str, dict[str, float]],
) -> list[dict[str, float]]:
    return [
        reliability_features(question, evidence_by_question.get(question.id, []), historical_probabilities, source_features)
        for question in questions
        if question.outcome is not None
    ]


def candidate_reliability_rules(feature_rows: list[dict[str, float]] | None = None) -> list[ReliabilityRule]:
    feature_rows = feature_rows or []
    if not feature_rows:
        min_hist_coverages = [0.5, 0.75]
        market_override_sharpnesses = [0.0, 0.05]
        prior_override_sharpnesses = [0.12, 0.18, 0.24]
        source_prior_route_sharpnesses = [0.18, 0.24, 0.30]
        noisy_source_strengths = [0.30, 0.32, 0.35, 0.40]
        max_hist_horizons = [150, 400, 9999]
        search_strengths = [0.50, 999.0]
        max_search_disagreements = [0.35]
    else:
        min_hist_coverages = _feature_thresholds(
            [row for row in feature_rows if row.get("hist_available", 0.0) > 0.5],
            "source_hist_coverage",
            [0.5],
            [0.50, 0.75],
            max_values=3,
        )
        market_override_sharpnesses = _feature_thresholds(
            [row for row in feature_rows if row.get("market_available", 0.0) > 0.5],
            "market_sharpness",
            [0.05],
            [0.50, 0.75],
            max_values=3,
        )
        prior_override_sharpnesses = _feature_thresholds(feature_rows, "base_prior_sharpness", [0.18], [0.50, 0.75], max_values=3)
        source_prior_route_sharpnesses = _feature_thresholds(feature_rows, "source_prior_sharpness", [0.18], [0.50, 0.75], max_values=3)
        noisy_source_strengths = _feature_thresholds(feature_rows, "source_avg_evidence_strength", [0.30], [0.50, 0.75], max_values=3)
        max_hist_horizons = [
            int(value)
            for value in _feature_thresholds(feature_rows, "horizon_days", [400], [0.50, 0.75], max_values=3)
        ]
        search_strengths = _feature_thresholds(feature_rows, "evidence_strength", [0.50, 999.0], [0.75], max_values=3)
        max_search_disagreements = _feature_thresholds(feature_rows, "evidence_disagreement", [0.35], [0.75], max_values=2)

    rules = []
    for min_hist_coverage in min_hist_coverages:
        for market_override_sharpness in market_override_sharpnesses:
            for prior_override_sharpness in prior_override_sharpnesses:
                for source_prior_route_sharpness in source_prior_route_sharpnesses:
                    for noisy_source_strength in noisy_source_strengths:
                        for max_hist_horizon_days in max_hist_horizons:
                            for search_strength in search_strengths:
                                for max_search_disagreement in max_search_disagreements:
                                    rules.append(
                                        ReliabilityRule(
                                            min_hist_coverage=min_hist_coverage,
                                            market_override_sharpness=market_override_sharpness,
                                            prior_override_sharpness=prior_override_sharpness,
                                            source_prior_route_sharpness=source_prior_route_sharpness,
                                            noisy_source_strength=noisy_source_strength,
                                            max_hist_horizon_days=max_hist_horizon_days,
                                            search_strength=search_strength,
                                            max_search_disagreement=max_search_disagreement,
                                        )
                                    )
    dedup: dict[tuple[object, ...], ReliabilityRule] = {}
    for rule in rules:
        key = tuple(rule.__dict__.items())
        dedup.setdefault(key, rule)
    return list(dedup.values())


def fit_reliability_rule(
    results: list[ForecastResult],
    questions: list[ForecastQuestion],
    evidence_by_question: dict[str, list[EvidenceItem]],
    historical_probabilities: dict[str, float],
    source_features: dict[str, dict[str, float]],
    candidate_systems: set[str],
    self_adjust_thresholds: bool = False,
) -> ReliabilityRule:
    by_pair = {(result.question_id, result.system): result for result in results if result.system in candidate_systems}
    scored_questions = [question for question in questions if question.outcome is not None]
    records: list[tuple[dict[str, float], dict[str, float], int]] = []
    for question in scored_questions:
        features = reliability_features(question, evidence_by_question.get(question.id, []), historical_probabilities, source_features)
        probabilities = {
            system: result.scored_probability()
            for system in candidate_systems
            if (result := by_pair.get((question.id, system))) is not None
        }
        if probabilities:
            records.append((features, probabilities, int(question.outcome)))

    def rule_loss(rule: ReliabilityRule) -> tuple[float, float]:
        brier_total = 0.0
        log_total = 0.0
        n = 0
        for features, probabilities, outcome in records:
            system = route_from_features(features, rule)
            probability = probabilities.get(system)
            if probability is None:
                continue
            brier_total += brier_score(probability, outcome)
            log_total += log_score(probability, outcome)
            n += 1
        return brier_total / max(1, n), log_total / max(1, n)

    feature_rows = [features for features, _, _ in records] if self_adjust_thresholds else None
    return min(candidate_reliability_rules(feature_rows), key=rule_loss)


def load_questions(raw_dir: Path, question_set: str, market_subset: bool) -> tuple[list[ForecastQuestion], dict[str, list[EvidenceItem]]]:
    question_file, resolution_file = download_forecastbench_files(raw_dir, question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, market_subset)
    return questions, evidence_by_question


def build_systems(questions: list[ForecastQuestion], cache_dir: Path, sources: list[str]) -> tuple[dict[str, ForecastingSystem], dict[str, float]]:
    probabilities = build_probabilities(questions, cache_dir, set(sources))
    provider = LocalEvidenceSearchProvider(probabilities)
    reasoner = HeuristicReasoner()
    systems: dict[str, ForecastingSystem] = {
        "aia_baseline": AIABaselineForecaster(),
        "market_only": MarketOnlyForecaster(),
        "evidence_graph_v1": EvidenceGraphForecaster(),
        "historical_analog": HistoricalAnalogForecaster(probabilities),
        "historical_evidence_blend": HistoricalEvidenceBlendForecaster(probabilities, analog_weight=1.0),
        "search_llm_loop": SearchEnabledLLMForecaster(search_provider=provider, reasoner=reasoner, evidence_weight=1.0, analog_gate=True),
        "hybrid_analog_search": HybridAnalogSearchForecaster(search_provider=provider, reasoner=reasoner, evidence_weight=1.0),
    }
    return systems, probabilities


def run_systems(systems: dict[str, ForecastingSystem], questions: list[ForecastQuestion], evidence_by_question: dict[str, list[EvidenceItem]]) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    for question in questions:
        evidence = evidence_by_question.get(question.id, [])
        for system in systems.values():
            results.append(system.forecast(question, evidence))
    return results


def source_system_metrics(results: list[ForecastResult], questions: list[ForecastQuestion]) -> list[dict[str, object]]:
    by_question = {question.id: question for question in questions}
    grouped: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for result in results:
        question = by_question.get(result.question_id)
        if question is None or question.outcome is None:
            continue
        grouped[(source_of(question), result.system)].append((result.scored_probability(), int(question.outcome)))
    rows = []
    for (source, system), scored in sorted(grouped.items()):
        rows.append(
            {
                "source": source,
                "system": system,
                "n": len(scored),
                "brier": sum(brier_score(p, y) for p, y in scored) / len(scored),
                "log_score": sum(log_score(p, y) for p, y in scored) / len(scored),
                "calibration_error": calibration_error(scored),
            }
        )
    return rows


def fit_router(source_rows: list[dict[str, object]], allowed_systems: list[str], min_n: int) -> tuple[dict[str, str], str]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_system_scores: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in source_rows:
        if row["system"] not in allowed_systems:
            continue
        by_source[str(row["source"])].append(row)
        by_system_scores[str(row["system"])].append((float(row["brier"]), int(row["n"])))
    default_system = min(
        allowed_systems,
        key=lambda system: sum(score * n for score, n in by_system_scores[system]) / max(1, sum(n for _, n in by_system_scores[system])),
    )
    mapping = {}
    for source, rows in by_source.items():
        n = max(int(row["n"]) for row in rows)
        if n < min_n:
            mapping[source] = default_system
        else:
            mapping[source] = str(min(rows, key=lambda row: (float(row["brier"]), float(row["log_score"])))["system"])
    return mapping, default_system


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, train_set: str, eval_set: str, source_map: dict[str, str], default_system: str, eval_summary: dict[str, dict[str, float]], eval_source_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Source-Routed Gate Transfer Experiment",
        "",
        f"Calibration set: `{train_set}`",
        f"Evaluation set: `{eval_set}`",
        f"Default system: `{default_system}`",
        "",
        "## Learned Source Router",
        "",
        "| Source | Selected system |",
        "|---|---|",
    ]
    for source, system in sorted(source_map.items()):
        lines.append(f"| {source} | `{system}` |")
    lines.extend(["", "## Evaluation Metrics", "", "| System | N | Brier | Log score | Calibration error |", "|---|---:|---:|---:|---:|"])
    for system, row in sorted(eval_summary.items(), key=lambda item: item[1]["brier"]):
        lines.append(f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |")
    selected = ["reliability_routed_gate", "taxonomy_routed_gate", "source_routed_gate", "historical_analog", "search_llm_loop", "market_only", "aia_baseline"]
    values = sorted({str(row["source"]) for row in eval_source_rows})
    by_pair = {(str(row["source"]), str(row["system"])): row for row in eval_source_rows}
    lines.extend(["", "## Evaluation Source Brier (N)", "", "| Source | " + " | ".join(selected) + " |", "|---|" + "---:|" * len(selected)])
    for source in values:
        cells = []
        for system in selected:
            row = by_pair.get((source, system))
            cells.append("" if row is None else f"{float(row['brier']):.4f} ({int(row['n'])})")
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    train_questions, train_evidence = load_questions(args.raw_dir, args.train_question_set, args.market_subset)
    eval_questions, eval_evidence = load_questions(args.raw_dir, args.eval_question_set, args.market_subset)
    train_systems, train_probabilities = build_systems(train_questions, args.cache_dir, args.sources)
    train_source_features = build_source_features(train_questions, train_evidence, train_probabilities)
    train_results = run_systems(train_systems, train_questions, train_evidence)
    train_source_rows = source_system_metrics(train_results, train_questions)
    allowed = ["aia_baseline", "market_only", "evidence_graph_v1", "historical_analog", "search_llm_loop"]
    source_map, default_system = fit_router(train_source_rows, allowed, args.min_source_n)
    reliability_rule = fit_reliability_rule(
        train_results,
        train_questions,
        train_evidence,
        train_probabilities,
        train_source_features,
        set(allowed),
        self_adjust_thresholds=args.self_adjust_thresholds,
    )

    eval_systems, eval_probabilities = build_systems(eval_questions, args.cache_dir, args.sources)
    eval_source_features = build_source_features(eval_questions, eval_evidence, eval_probabilities)
    eval_systems["source_routed_gate"] = SourceRoutedForecaster(eval_systems, source_map, default_system)
    eval_systems["taxonomy_routed_gate"] = TaxonomyRoutedForecaster(eval_systems)
    eval_systems["reliability_routed_gate"] = ReliabilityRoutedForecaster(eval_systems, eval_probabilities, eval_source_features, reliability_rule)
    eval_results = run_systems(eval_systems, eval_questions, eval_evidence)
    outcomes = {question.id: int(question.outcome) for question in eval_questions if question.outcome is not None}
    eval_summary = summarize_results(eval_results, outcomes)
    eval_source_rows = source_system_metrics(eval_results, eval_questions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "source_router.json").write_text(
        json.dumps(
            {
                "mapping": source_map,
                "default_system": default_system,
                "reliability_rule": reliability_rule.__dict__,
                "self_adjust_thresholds": args.self_adjust_thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "eval_metrics.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    (args.out_dir / "eval_results.json").write_text(json.dumps([result_to_dict(result) for result in eval_results], indent=2), encoding="utf-8")
    write_csv(args.out_dir / "train_source_metrics.csv", train_source_rows)
    write_csv(args.out_dir / "eval_source_metrics.csv", eval_source_rows)
    write_report(args.out_dir, args.train_question_set, args.eval_question_set, source_map, default_system, eval_summary, eval_source_rows)
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-question-set", default="2024-07-21-human")
    parser.add_argument("--eval-question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/source_routed_gate_transfer_20260626"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics", "yfinance"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--min-source-n", type=int, default=8)
    parser.add_argument(
        "--self-adjust-thresholds",
        action="store_true",
        help="Add calibration-distribution quantiles to the ReliabilityRoute threshold grid before selecting the frozen rule.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
