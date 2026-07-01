from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .fixtures import SAMPLE_EVIDENCE, SAMPLE_QUESTIONS
from .forecasters import (
    AIABaselineForecaster,
    AdvancedForecastLab,
    AdvancedNoCalibrationForecastLab,
    AdvancedNoMarketForecastLab,
    AdvancedNoMemoryForecastLab,
    EvidenceGraphForecaster,
    EvidenceGraphNoMarketForecaster,
    MarketOnlyForecaster,
)
from .metrics import summarize_results
from .models import EvidenceItem, ForecastQuestion, ForecastResult
from .synthetic_benchmark import generate_synthetic_dataset, write_dataset


FORECASTERS = {
    "market_only": MarketOnlyForecaster,
    "aia_baseline": AIABaselineForecaster,
    "evidence_graph_no_market": EvidenceGraphNoMarketForecaster,
    "evidence_graph_v1": EvidenceGraphForecaster,
    "advanced_no_market": AdvancedNoMarketForecastLab,
    "advanced_no_memory": AdvancedNoMemoryForecastLab,
    "advanced_no_calibration": AdvancedNoCalibrationForecastLab,
    "advanced_v1": AdvancedForecastLab,
}


def question_from_dict(row: dict[str, Any]) -> ForecastQuestion:
    return ForecastQuestion(
        id=row["id"],
        title=row["title"],
        description=row.get("description", ""),
        resolution_criteria=row.get("resolution_criteria", ""),
        domain=row.get("domain", "default"),
        cutoff_date=row.get("cutoff_date", ""),
        resolution_date=row.get("resolution_date", ""),
        outcome=row.get("outcome"),
        market_probability=row.get("market_probability"),
        metadata=row.get("metadata", {}),
    )


def evidence_from_dict(row: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        id=row["id"],
        source=row.get("source", "unknown"),
        text=row.get("text", ""),
        stance=row.get("stance", "neutral"),
        weight=float(row.get("weight", 0.0)),
        date=row.get("date"),
        tags=tuple(row.get("tags", [])),
    )


def load_dataset(path: Path | None) -> tuple[list[ForecastQuestion], dict[str, list[EvidenceItem]]]:
    if path is None:
        return SAMPLE_QUESTIONS, SAMPLE_EVIDENCE
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [question_from_dict(row) for row in data["questions"]]
    evidence = {
        question_id: [evidence_from_dict(item) for item in items]
        for question_id, items in data.get("evidence", {}).items()
    }
    return questions, evidence


def result_to_dict(result: ForecastResult) -> dict[str, Any]:
    return {
        "question_id": result.question_id,
        "system": result.system,
        "probability": round(result.scored_probability(), 6),
        "confidence": round(result.confidence, 6),
        "rationale": result.rationale,
        "diagnostics": result.diagnostics,
        "component_forecasts": [
            {
                "agent": component.agent,
                "probability": round(component.probability, 6),
                "confidence": round(component.confidence, 6),
                "rationale": component.rationale,
                "evidence_ids": list(component.evidence_ids),
            }
            for component in result.component_forecasts
        ],
        "subquestions": [
            {"id": sub.id, "text": sub.text, "weight": sub.weight, "prior": sub.prior}
            for sub in result.subquestions
        ],
    }


def grouped_summary(
    results: list[ForecastResult],
    questions: list[ForecastQuestion],
    group_key: str,
) -> dict[str, dict[str, dict[str, float]]]:
    question_by_id = {question.id: question for question in questions}
    output: dict[str, dict[str, dict[str, float]]] = {}
    for result in results:
        question = question_by_id.get(result.question_id)
        if question is None or question.outcome is None:
            continue
        label = group_label(question, group_key)
        output.setdefault(label, {})
    for label in sorted(output):
        label_ids = {q.id for q in questions if group_label(q, group_key) == label and q.outcome is not None}
        label_results = [result for result in results if result.question_id in label_ids]
        outcomes = {q.id: int(q.outcome) for q in questions if q.id in label_ids and q.outcome is not None}
        output[label] = summarize_results(label_results, outcomes)
    return output


def group_label(question: ForecastQuestion, group_key: str) -> str:
    if group_key == "domain":
        return question.domain
    if group_key == "liquidity":
        liquidity = float(question.metadata.get("market_liquidity", 0.5))
        if liquidity < 0.33:
            return "low_liquidity"
        if liquidity > 0.67:
            return "high_liquidity"
        return "mid_liquidity"
    if group_key == "horizon":
        horizon = int(question.metadata.get("horizon_days", 120))
        if horizon <= 60:
            return "short_horizon"
        if horizon >= 240:
            return "long_horizon"
        return "mid_horizon"
    raise ValueError(group_key)


def write_rows_csv(results: list[ForecastResult], questions: list[ForecastQuestion], out_dir: Path) -> None:
    question_by_id = {question.id: question for question in questions}
    fields = [
        "question_id",
        "system",
        "domain",
        "outcome",
        "probability",
        "confidence",
        "market_probability",
        "market_liquidity",
        "horizon_days",
        "latent_probability",
    ]
    with (out_dir / "forecast_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            question = question_by_id[result.question_id]
            writer.writerow(
                {
                    "question_id": question.id,
                    "system": result.system,
                    "domain": question.domain,
                    "outcome": question.outcome,
                    "probability": f"{result.scored_probability():.6f}",
                    "confidence": f"{result.confidence:.6f}",
                    "market_probability": "" if question.market_probability is None else f"{question.market_probability:.6f}",
                    "market_liquidity": question.metadata.get("market_liquidity", ""),
                    "horizon_days": question.metadata.get("horizon_days", ""),
                    "latent_probability": question.metadata.get("latent_probability", ""),
                }
            )


def write_report(
    results: list[ForecastResult],
    questions: list[ForecastQuestion],
    summary: dict[str, dict[str, float]],
    out_dir: Path,
    max_forecast_lines: int,
) -> None:
    domain_summary = grouped_summary(results, questions, "domain")
    liquidity_summary = grouped_summary(results, questions, "liquidity")
    lines = [
        "# Forecasting Agent Experiment",
        "",
        "This offline harness compares an AIA-style baseline against evidence-graph and advanced forecast-lab prototypes. Synthetic runs use a documented latent data generator; they are architecture stress tests, not external benchmark claims.",
        "",
        "## Metrics",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in sorted(summary.items()):
        lines.append(
            f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |"
        )
    lines.extend(["", "## Domain Breakdown", ""])
    for domain, rows in sorted(domain_summary.items()):
        lines.extend([f"### {domain}", "", "| System | N | Brier | Log score | Calibration error |", "|---|---:|---:|---:|---:|"])
        for system, row in sorted(rows.items()):
            lines.append(f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |")
        lines.append("")
    lines.extend(["## Liquidity Breakdown", ""])
    for bucket, rows in sorted(liquidity_summary.items()):
        lines.extend([f"### {bucket}", "", "| System | N | Brier | Log score | Calibration error |", "|---|---:|---:|---:|---:|"])
        for system, row in sorted(rows.items()):
            lines.append(f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |")
        lines.append("")
    lines.extend(["", "## Forecasts", ""])
    for result in results[:max_forecast_lines]:
        lines.append(
            f"- `{result.system}` on `{result.question_id}`: p={result.scored_probability():.3f}, "
            f"confidence={result.confidence:.3f}; {result.rationale}"
        )
    if len(results) > max_forecast_lines:
        lines.append(f"- ... {len(results) - max_forecast_lines} additional forecast rows omitted from Markdown; see `forecast_rows.csv` and `results.json`.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.synthetic_size:
        questions, evidence_by_question = generate_synthetic_dataset(args.synthetic_size, args.synthetic_seed)
    else:
        questions, evidence_by_question = load_dataset(args.dataset)
    selected = args.systems or list(FORECASTERS)
    forecasters = [FORECASTERS[name]() for name in selected]
    run_name = args.run_name or time.strftime("forecasting_%Y%m%d-%H%M%S")
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_size:
        write_dataset(out_dir / "synthetic_dataset.json", questions, evidence_by_question)
    results: list[ForecastResult] = []
    for question in questions[: args.limit or None]:
        evidence = evidence_by_question.get(question.id, [])
        for forecaster in forecasters:
            result = forecaster.forecast(question, evidence)
            results.append(result)
            if not args.quiet:
                print(f"{result.system} {question.id} p={result.scored_probability():.3f} confidence={result.confidence:.3f}")
    outcomes = {q.id: int(q.outcome) for q in questions if q.outcome is not None}
    summary = summarize_results(results, outcomes)
    (out_dir / "results.json").write_text(
        json.dumps([result_to_dict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_rows_csv(results, questions, out_dir)
    write_report(results, questions, summary, out_dir, args.max_forecast_lines)
    print(out_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--systems", nargs="+", choices=sorted(FORECASTERS), default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic-size", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=7)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-forecast-lines", type=int, default=80)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
