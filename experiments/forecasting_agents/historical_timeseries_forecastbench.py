from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .forecastbench_adapter import download_forecastbench_files, load_forecastbench_targets
from .forecasters import AIABaselineForecaster, EvidenceGraphForecaster
from .metrics import brier_score, summarize_results
from .models import ForecastQuestion, ForecastResult, clamp_probability
from .run_forecasting_experiment import result_to_dict


class HistoricalAnalogForecaster:
    name = "historical_analog"

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def forecast(self, question: ForecastQuestion, evidence: list[Any]) -> ForecastResult:
        probability = self.probabilities.get(question.id, question.metadata.get("base_prior", 0.5))
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=float(probability),
            confidence=0.65 if question.id in self.probabilities else 0.30,
            rationale="Historical analog rate: past pre-cutoff observations were compared to values at the same horizon.",
            diagnostics={"historical_analog_available": question.id in self.probabilities},
        )


class HistoricalEvidenceBlendForecaster:
    name = "historical_evidence_blend"

    def __init__(self, probabilities: dict[str, float], analog_weight: float = 1.0) -> None:
        self.probabilities = probabilities
        self.analog_weight = analog_weight
        self.graph = EvidenceGraphForecaster()

    def forecast(self, question: ForecastQuestion, evidence: list[Any]) -> ForecastResult:
        graph_result = self.graph.forecast(question, evidence)
        analog_probability = self.probabilities.get(question.id)
        if analog_probability is None:
            probability = graph_result.probability
        else:
            probability = self.analog_weight * analog_probability + (1.0 - self.analog_weight) * graph_result.probability
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=probability,
            confidence=graph_result.confidence,
            rationale="Blends historical analog evidence with the evidence graph when leakage-checked history is available.",
            component_forecasts=graph_result.component_forecasts,
            evidence=graph_result.evidence,
            subquestions=graph_result.subquestions,
            diagnostics={**graph_result.diagnostics, "historical_analog_available": analog_probability is not None, "analog_weight": self.analog_weight},
        )


def fetch_fred_history(series_id: str, cache_dir: Path) -> list[tuple[datetime, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"fred_{series_id}.csv"
    if not path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    rows: list[tuple[datetime, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = _float_or_none(row.get(series_id))
            if value is None:
                continue
            rows.append((datetime.fromisoformat(row["observation_date"]), value))
    return rows


def fetch_dbnomics_history(series_id: str, cache_dir: Path) -> list[tuple[datetime, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"dbnomics_{series_id.replace('/', '_').replace('.', '_')}.json"
    if not path.exists():
        provider, dataset, code = parse_dbnomics_id(series_id)
        url = f"https://api.db.nomics.world/v22/series/{provider}/{dataset}/{code}?observations=1"
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data.get("series", {}).get("docs", [])
    if not docs:
        return []
    periods = docs[0].get("period", [])
    values = docs[0].get("value", [])
    rows: list[tuple[datetime, float]] = []
    for period, value in zip(periods, values):
        parsed = _float_or_none(value)
        if parsed is None:
            continue
        rows.append((datetime.fromisoformat(str(period)[:10]), parsed))
    return rows


def fetch_yfinance_history(ticker: str, cache_dir: Path) -> list[tuple[datetime, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "-")
    path = cache_dir / f"yfinance_{safe_ticker}.json"
    if not path.exists():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=10y&interval=1d"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return parse_yfinance_chart(path.read_text(encoding="utf-8"))


def parse_yfinance_chart(payload: str) -> list[tuple[datetime, float]]:
    data = json.loads(payload)
    result = data.get("chart", {}).get("result", [])
    if not result:
        return []
    timestamps = result[0].get("timestamp", [])
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
    rows: list[tuple[datetime, float]] = []
    for timestamp, close in zip(timestamps, closes):
        value = _float_or_none(close)
        if value is None:
            continue
        rows.append((datetime.fromtimestamp(int(timestamp), timezone.utc).replace(tzinfo=None), value))
    return rows


def parse_dbnomics_id(series_id: str) -> tuple[str, str, str]:
    parts = series_id.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse DBnomics id: {series_id}")
    return parts[0], parts[1], parts[2]


def historical_probability(history: list[tuple[datetime, float]], cutoff: str, resolution_date: str, current_value: float | None) -> float | None:
    if not history or not cutoff or not resolution_date:
        return None
    cutoff_dt = datetime.fromisoformat(cutoff[:10])
    resolution_dt = datetime.fromisoformat(resolution_date[:10])
    horizon_days = max(1, (resolution_dt - cutoff_dt).days)
    filtered = [(date, value) for date, value in history if date <= cutoff_dt]
    if len(filtered) < 20:
        return None
    dates = [date for date, _ in filtered]
    outcomes: list[int] = []
    for idx, (date, value) in enumerate(filtered):
        target_date = date + timedelta(days=horizon_days)
        target_idx = bisect_left(dates, target_date)
        if target_idx >= len(filtered):
            continue
        if (filtered[target_idx][0] - target_date).days > max(7, horizon_days * 0.15):
            continue
        outcomes.append(1 if filtered[target_idx][1] > value else 0)
    if len(outcomes) < 20:
        return None
    analog_rate = sum(outcomes) / len(outcomes)
    if current_value is None:
        return clamp_probability(analog_rate)
    values = [value for _, value in filtered]
    percentile = sum(1 for value in values if value <= current_value) / len(values)
    mean_reversion_adjustment = (0.5 - percentile) * 0.18
    return clamp_probability(0.82 * analog_rate + 0.18 * 0.5 + mean_reversion_adjustment)


def build_probabilities(questions: list[ForecastQuestion], cache_dir: Path, sources: set[str]) -> dict[str, float]:
    histories: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    probabilities: dict[str, float] = {}
    for question in questions:
        source = str(question.metadata.get("forecastbench_source", question.domain))
        if source not in sources:
            continue
        series_id = str(question.metadata.get("forecastbench_base_id", ""))
        key = (source, series_id)
        if key not in histories:
            try:
                if source == "fred":
                    histories[key] = fetch_fred_history(series_id, cache_dir)
                elif source == "dbnomics":
                    histories[key] = fetch_dbnomics_history(series_id, cache_dir)
                elif source == "yfinance":
                    histories[key] = fetch_yfinance_history(series_id, cache_dir)
                else:
                    histories[key] = []
            except Exception:
                histories[key] = []
        current_value = _float_or_none(question.metadata.get("freeze_datetime_value"))
        probability = historical_probability(histories[key], str(question.metadata.get("forecast_due_date", "")), question.resolution_date, current_value)
        if probability is not None:
            probabilities[question.id] = probability
    return probabilities


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


def run(args: argparse.Namespace) -> None:
    question_file, resolution_file = download_forecastbench_files(args.raw_dir, args.question_set)
    questions, evidence_by_question, _ = load_forecastbench_targets(question_file, resolution_file, args.market_subset)
    if args.limit:
        questions = questions[: args.limit]
    probabilities = build_probabilities(questions, args.cache_dir, set(args.sources))
    systems = [
        AIABaselineForecaster(),
        EvidenceGraphForecaster(),
        HistoricalAnalogForecaster(probabilities),
        HistoricalEvidenceBlendForecaster(probabilities, analog_weight=args.analog_weight),
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
    (args.out_dir / "historical_probabilities.json").write_text(json.dumps(probabilities, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps([result_to_dict(result) for result in results], indent=2), encoding="utf-8")
    write_rows(results, outcomes, args.out_dir)
    write_report(args.out_dir, args.question_set, len(questions), len(probabilities), summary)
    print(args.out_dir / "summary.md")


def write_report(out_dir: Path, question_set: str, targets: int, covered: int, summary: dict[str, dict[str, float]]) -> None:
    lines = [
        "# ForecastBench Historical Time-Series Analog",
        "",
        f"Question set: `{question_set}`",
        f"Targets: `{targets}`",
        f"Historical analog coverage: `{covered}` targets",
        "",
        "Historical analog forecasts are computed from source histories using only observations dated no later than the forecast due date.",
        "",
        "| System | N | Brier | Log score | Calibration error |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, row in sorted(summary.items(), key=lambda item: item[1]["brier"]):
        lines.append(f"| {system} | {int(row['n'])} | {row['brier']:.4f} | {row['log_score']:.4f} | {row['calibration_error']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-set", default="2025-10-26-llm")
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/forecastbench_data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/forecastbench_data/timeseries_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/forecastbench_historical_timeseries_20260625"))
    parser.add_argument("--sources", nargs="+", default=["fred", "dbnomics"], choices=["fred", "dbnomics", "yfinance"])
    parser.add_argument("--analog-weight", type=float, default=1.0)
    parser.add_argument("--market-subset", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
