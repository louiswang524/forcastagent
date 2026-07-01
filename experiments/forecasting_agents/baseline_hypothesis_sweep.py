from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import AIABaselineForecaster, EvidenceGraphForecaster
from .historical_timeseries_forecastbench import HistoricalAnalogForecaster, HistoricalEvidenceBlendForecaster, build_probabilities
from .metrics import brier_score, calibration_error, log_score
from .models import ForecastQuestion, ForecastResult
from .search_llm_forecaster import HeuristicReasoner, LocalEvidenceSearchProvider, SearchEnabledLLMForecaster


@dataclass(frozen=True)
class Candidate:
    hypothesis: str
    system: str
    candidate_id: str
    sources: tuple[str, ...]
    analog_weight: float | None = None
    search_weight: float | None = None


@dataclass(frozen=True)
class Dataset:
    questions: list[ForecastQuestion]
    evidence_by_question: dict[str, list]
    outcomes: dict[str, int]


def load_dataset(raw_dir: Path, question_set: str, market_subset: bool) -> Dataset:
    question_file, resolution_file = download_forecastbench_files(raw_dir, question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, market_subset)
    outcomes = {question.id: int(question.outcome) for question in questions if question.outcome is not None}
    return Dataset(questions=questions, evidence_by_question=evidence_by_question, outcomes=outcomes)


def source_policies() -> list[tuple[str, ...]]:
    return [
        ("fred",),
        ("dbnomics",),
        ("yfinance",),
        ("fred", "dbnomics"),
        ("fred", "yfinance"),
        ("dbnomics", "yfinance"),
        ("fred", "dbnomics", "yfinance"),
    ]


def build_candidates() -> list[Candidate]:
    candidates = [
        Candidate("H26_static_baselines", "aia_baseline", "aia_baseline", ()),
        Candidate("H26_static_baselines", "evidence_graph_v1", "evidence_graph_v1", ()),
    ]
    for sources in source_policies():
        source_id = "+".join(sources)
        candidates.append(Candidate("H27_source_policy", "historical_analog", f"historical_analog__{source_id}", sources))
        for weight in [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]:
            candidates.append(
                Candidate(
                    "H28_analog_blend_weight",
                    "historical_evidence_blend",
                    f"historical_evidence_blend__{source_id}__aw{weight:.2f}",
                    sources,
                    analog_weight=weight,
                )
            )
        for weight in [0.0, 0.25, 0.50, 0.62, 0.75, 1.0]:
            candidates.append(
                Candidate(
                    "H29_search_graph_weight",
                    "search_llm_loop",
                    f"search_llm_loop__{source_id}__sw{weight:.2f}",
                    sources,
                    search_weight=weight,
                )
            )
    return candidates


def instantiate(candidate: Candidate, probabilities: dict[str, float]):
    if candidate.system == "aia_baseline":
        return AIABaselineForecaster()
    if candidate.system == "evidence_graph_v1":
        return EvidenceGraphForecaster()
    if candidate.system == "historical_analog":
        return HistoricalAnalogForecaster(probabilities)
    if candidate.system == "historical_evidence_blend":
        return HistoricalEvidenceBlendForecaster(probabilities, analog_weight=float(candidate.analog_weight))
    if candidate.system == "search_llm_loop":
        return SearchEnabledLLMForecaster(
            search_provider=LocalEvidenceSearchProvider(probabilities),
            reasoner=HeuristicReasoner(),
            evidence_weight=float(candidate.search_weight),
        )
    raise ValueError(f"Unsupported candidate system: {candidate.system}")


def evaluate(candidate: Candidate, dataset: Dataset, probabilities: dict[str, float]) -> dict[str, float | str | int]:
    forecaster = instantiate(candidate, probabilities)
    results: list[ForecastResult] = []
    for question in dataset.questions:
        evidence = dataset.evidence_by_question.get(question.id, [])
        result = forecaster.forecast(question, evidence)
        results.append(result)
    scored = [
        (result.scored_probability(), dataset.outcomes[result.question_id])
        for result in results
        if result.question_id in dataset.outcomes
    ]
    return {
        "hypothesis": candidate.hypothesis,
        "system": candidate.system,
        "candidate_id": candidate.candidate_id,
        "sources": "+".join(candidate.sources) if candidate.sources else "none",
        "analog_weight": "" if candidate.analog_weight is None else f"{candidate.analog_weight:.2f}",
        "search_weight": "" if candidate.search_weight is None else f"{candidate.search_weight:.2f}",
        "n": len(scored),
        "coverage": len(probabilities),
        "brier": sum(brier_score(probability, outcome) for probability, outcome in scored) / len(scored),
        "log_score": sum(log_score(probability, outcome) for probability, outcome in scored) / len(scored),
        "calibration_error": calibration_error(scored),
    }


def probability_cache(
    dataset: Dataset,
    candidates: list[Candidate],
    cache_dir: Path,
) -> dict[tuple[str, ...], dict[str, float]]:
    by_sources: dict[tuple[str, ...], dict[str, float]] = {}
    for candidate in candidates:
        if not candidate.sources or candidate.sources in by_sources:
            continue
        by_sources[candidate.sources] = build_probabilities(dataset.questions, cache_dir, set(candidate.sources))
    by_sources[()] = {}
    return by_sources


def select_transfer_rows(train_rows: list[dict], eval_rows: list[dict]) -> list[dict]:
    eval_by_candidate = {str(row["candidate_id"]): row for row in eval_rows}
    selected = []
    for hypothesis in sorted({str(row["hypothesis"]) for row in train_rows}):
        hypothesis_rows = [row for row in train_rows if row["hypothesis"] == hypothesis]
        best_train = min(hypothesis_rows, key=lambda row: (float(row["brier"]), float(row["log_score"])))
        eval_row = eval_by_candidate[str(best_train["candidate_id"])]
        selected.append(
            {
                "hypothesis": hypothesis,
                "selected_candidate": best_train["candidate_id"],
                "system": best_train["system"],
                "sources": best_train["sources"],
                "analog_weight": best_train["analog_weight"],
                "search_weight": best_train["search_weight"],
                "train_brier": best_train["brier"],
                "eval_brier": eval_row["brier"],
                "train_log_score": best_train["log_score"],
                "eval_log_score": eval_row["log_score"],
                "eval_calibration_error": eval_row["calibration_error"],
                "eval_coverage": eval_row["coverage"],
            }
        )
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_dir: Path, train_set: str, eval_set: str, train_rows: list[dict], eval_rows: list[dict], transfer_rows: list[dict]) -> None:
    best_eval = sorted(eval_rows, key=lambda row: (float(row["brier"]), float(row["log_score"])))[:10]
    lines = [
        "# Baseline Hypothesis Sweep",
        "",
        f"Calibration set: `{train_set}`",
        f"Evaluation set: `{eval_set}`",
        "",
        "## Hypotheses",
        "",
        "- H26: The implemented AIA-style baseline is weak mainly because static evidence aggregation underuses structured historical analog evidence.",
        "- H27: Retrieval source policy matters; enabling every historical source can hurt when source-specific parsers are noisy.",
        "- H28: Analog/evidence blend weight should be selected on a calibration vintage rather than fixed globally.",
        "- H29: Search-loop graph/evidence blend weight should be selected on a calibration vintage rather than fixed globally.",
        "",
        "## No-Retune Transfer Selections",
        "",
        "| Hypothesis | Selected candidate | Train Brier | Eval Brier | Eval Log | Eval Cal error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in transfer_rows:
        lines.append(
            f"| {row['hypothesis']} | `{row['selected_candidate']}` | {float(row['train_brier']):.4f} | "
            f"{float(row['eval_brier']):.4f} | {float(row['eval_log_score']):.4f} | {float(row['eval_calibration_error']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Evaluation Candidates",
            "",
            "| Rank | Candidate | Hypothesis | Eval Brier | Eval Log | Sources | Coverage |",
            "|---:|---|---|---:|---:|---|---:|",
        ]
    )
    for rank, row in enumerate(best_eval, start=1):
        lines.append(
            f"| {rank} | `{row['candidate_id']}` | {row['hypothesis']} | {float(row['brier']):.4f} | "
            f"{float(row['log_score']):.4f} | {row['sources']} | {int(row['coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The transfer table is the main result. The best evaluation table is diagnostic only because it uses evaluation labels to rank candidates.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    candidates = build_candidates()
    train = load_dataset(args.raw_dir, args.train_question_set, args.market_subset)
    eval_set = load_dataset(args.raw_dir, args.eval_question_set, args.market_subset)
    train_probabilities = probability_cache(train, candidates, args.cache_dir)
    eval_probabilities = probability_cache(eval_set, candidates, args.cache_dir)
    train_rows = [evaluate(candidate, train, train_probabilities[candidate.sources]) for candidate in candidates]
    eval_rows = [evaluate(candidate, eval_set, eval_probabilities[candidate.sources]) for candidate in candidates]
    transfer_rows = select_transfer_rows(train_rows, eval_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "train_candidates.csv", train_rows)
    write_csv(args.out_dir / "eval_candidates.csv", eval_rows)
    write_csv(args.out_dir / "transfer_selections.csv", transfer_rows)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "train_question_set": args.train_question_set,
                "eval_question_set": args.eval_question_set,
                "market_subset": args.market_subset,
                "candidate_count": len(candidates),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(args.out_dir, args.train_question_set, args.eval_question_set, train_rows, eval_rows, transfer_rows)
    print(args.out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-question-set", default="2024-07-21-human")
    parser.add_argument("--eval-question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/baseline_hypothesis_sweep_20260626"))
    parser.add_argument("--market-subset", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
