from __future__ import annotations

import json
import statistics
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import EvidenceItem, ForecastQuestion


MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}


QUESTION_URL = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/question_sets/{name}"
RESOLUTION_URL = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/resolution_sets/{name}"
SUPER_URL = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/forecast_sets/2024-07-21/2024-07-21.ForecastBench.human_super_individual.json"
PUBLIC_URL = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/forecast_sets/2024-07-21/2024-07-21.ForecastBench.human_public_individual.json"


def download_forecastbench_files(raw_dir: Path, question_set: str) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    question_file = raw_dir / f"{question_set}.json"
    date = question_set.replace("-human", "").replace("-llm", "")
    resolution_file = raw_dir / f"{date}_resolution_set.json"
    if not question_file.exists():
        _download(QUESTION_URL.format(name=question_file.name), question_file)
    if not resolution_file.exists():
        _download(RESOLUTION_URL.format(name=resolution_file.name), resolution_file)
    return question_file, resolution_file


def download_superforecaster_file(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "2024-07-21.ForecastBench.human_super_individual.json"
    if not path.exists():
        _download(SUPER_URL, path)
    return path


def download_public_forecaster_file(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "2024-07-21.ForecastBench.human_public_individual.json"
    if not path.exists():
        _download(PUBLIC_URL, path)
    return path


def _download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def load_forecastbench_targets(
    question_file: Path,
    resolution_file: Path,
    market_subset: bool = False,
) -> tuple[list[ForecastQuestion], dict[str, list[EvidenceItem]], dict[str, str]]:
    question_data = json.loads(question_file.read_text(encoding="utf-8"))
    resolution_data = json.loads(resolution_file.read_text(encoding="utf-8"))
    questions_by_id = {
        row["id"]: row
        for row in question_data["questions"]
        if isinstance(row.get("id"), str)
    }
    targets: list[ForecastQuestion] = []
    evidence_by_target: dict[str, list[EvidenceItem]] = {}
    target_to_base_id: dict[str, str] = {}
    forecast_due_date = question_data.get("forecast_due_date", "")
    for row in resolution_data["resolutions"]:
        base_id = row.get("id")
        if not isinstance(base_id, str) or base_id not in questions_by_id:
            continue
        if not row.get("resolved") or row.get("resolved_to") not in (0, 1, 0.0, 1.0):
            continue
        question = questions_by_id[base_id]
        if market_subset and question.get("source") not in MARKET_SOURCES:
            continue
        resolution_date = row.get("resolution_date")
        target_id = target_key(base_id, resolution_date)
        market_probability = parse_market_probability(question)
        prior = horizon_adjusted_prior(question, forecast_due_date, resolution_date)
        rendered_question = render_text(question.get("question", ""), forecast_due_date, resolution_date)
        rendered_criteria = render_text(question.get("resolution_criteria", ""), forecast_due_date, resolution_date)
        target = ForecastQuestion(
            id=target_id,
            title=rendered_question,
            description=render_text(question.get("background", ""), forecast_due_date, resolution_date),
            resolution_criteria=rendered_criteria,
            domain=question.get("source", "forecastbench"),
            cutoff_date=question.get("freeze_datetime", forecast_due_date),
            resolution_date=resolution_date or "",
            outcome=int(row["resolved_to"]),
            market_probability=market_probability,
            metadata={
                "forecastbench_base_id": base_id,
                "forecastbench_source": question.get("source"),
                "forecast_due_date": forecast_due_date,
                "url": question.get("url"),
                "freeze_datetime_value": question.get("freeze_datetime_value"),
                "question_set": question_data.get("question_set"),
                "base_prior": prior,
                "horizon_bucket": horizon_bucket(forecast_due_date, resolution_date),
            },
        )
        targets.append(target)
        evidence_by_target[target_id] = forecastbench_evidence(target, question, market_probability)
        target_to_base_id[target_id] = base_id
    return targets, evidence_by_target, target_to_base_id


def parse_market_probability(question: dict[str, Any]) -> float | None:
    source = question.get("source")
    explanation = str(question.get("freeze_datetime_value_explanation", "")).lower()
    if source not in MARKET_SOURCES and "market" not in explanation and "crowd" not in explanation and "community prediction" not in explanation:
        return None
    try:
        value = float(question.get("freeze_datetime_value"))
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def source_base_prior(question: dict[str, Any]) -> float:
    source = str(question.get("source", ""))
    text = (str(question.get("question", "")) + " " + str(question.get("background", ""))).lower()
    if source == "acled":
        if "ten times" in text:
            return 0.07
        return 0.18
    if source == "yfinance":
        return 0.58
    if source == "wikipedia":
        return 0.28
    if source == "fred":
        return 0.38
    if source == "dbnomics":
        return 0.36
    if source in MARKET_SOURCES:
        market = parse_market_probability(question)
        return 0.5 if market is None else market
    return 0.5


def horizon_adjusted_prior(question: dict[str, Any], forecast_due_date: str, resolution_date: str | None) -> float:
    base = source_base_prior(question)
    bucket = horizon_bucket(forecast_due_date, resolution_date)
    source = str(question.get("source", ""))
    table = {
        ("acled", "near"): 0.06,
        ("acled", "short"): 0.07,
        ("acled", "mid"): 0.08,
        ("dbnomics", "near"): 0.48,
        ("dbnomics", "short"): 0.16,
        ("dbnomics", "mid"): 0.30,
        ("fred", "near"): 0.36,
        ("fred", "short"): 0.30,
        ("fred", "mid"): 0.42,
        ("wikipedia", "near"): 0.26,
        ("wikipedia", "short"): 0.22,
        ("wikipedia", "mid"): 0.20,
        ("yfinance", "near"): 0.55,
        ("yfinance", "short"): 0.76,
        ("yfinance", "mid"): 0.64,
    }
    adjusted = table.get((source, bucket), base)
    # Blend with base prior to keep this feature conservative.
    return 0.65 * adjusted + 0.35 * base


def horizon_bucket(forecast_due_date: str, resolution_date: str | None) -> str:
    if not forecast_due_date or not resolution_date:
        return "unknown"
    try:
        start = datetime.fromisoformat(forecast_due_date[:10])
        end = datetime.fromisoformat(resolution_date[:10])
    except ValueError:
        return "unknown"
    days = (end - start).days
    if days <= 45:
        return "near"
    if days <= 120:
        return "short"
    if days <= 400:
        return "mid"
    return "long"


def forecastbench_evidence(
    target: ForecastQuestion,
    question: dict[str, Any],
    market_probability: float | None,
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    forecast_due_date = str(target.metadata.get("forecast_due_date", ""))
    prior = horizon_adjusted_prior(question, forecast_due_date, target.resolution_date)
    if market_probability is None or question.get("source") not in MARKET_SOURCES:
        evidence.append(
            EvidenceItem(
                id=f"{target.id}-source_prior",
                source="forecastbench_source_prior",
                text=f"Source/type base prior for {question.get('source')}: {prior:.2f}.",
                stance="supports" if prior > 0.5 else "opposes",
                weight=min(0.45, abs(prior - 0.5) * 1.6),
                tags=("base_rate",),
            )
        )
    if market_probability is not None:
        evidence.append(
            EvidenceItem(
                id=f"{target.id}-market",
                source="forecastbench_market",
                text=f"ForecastBench freeze-time market/crowd probability: {market_probability:.4f}.",
                stance="supports" if market_probability >= 0.5 else "opposes",
                weight=min(0.35, abs(market_probability - 0.5) * 1.2),
                tags=("market", "recent"),
            )
        )
    source = str(question.get("source", ""))
    text = (question.get("question", "") + " " + question.get("background", "")).lower()
    if source == "acled" and "ten times" in text:
        evidence.append(_item(target.id, "base_rate", "High-threshold ACLED event wording usually implies a low base rate.", "opposes", 0.16, "base_rate"))
    elif source in {"fred", "dbnomics", "yfinance", "wikipedia"}:
        evidence.append(_item(target.id, "base_rate", f"{source} time-series direction question has no explicit market price; use weak 50/50 drift prior.", "neutral", 0.02, "base_rate"))
    if "before" in text or "by " in text:
        evidence.append(_item(target.id, "resolution", "Deadline wording makes timing part of resolution.", "opposes", 0.04, "resolution"))
    evidence.extend(structured_time_series_evidence(target, question))
    bucket = horizon_bucket(forecast_due_date, target.resolution_date)
    if bucket != "unknown":
        horizon_prior = horizon_adjusted_prior(question, forecast_due_date, target.resolution_date)
        source_prior = source_base_prior(question)
        delta = horizon_prior - source_prior
        if abs(delta) >= 0.03:
            evidence.append(
                EvidenceItem(
                    id=f"{target.id}-horizon",
                    source="forecastbench_horizon_prior",
                    text=f"Resolution horizon bucket {bucket} adjusts source prior from {source_prior:.2f} to {horizon_prior:.2f}.",
                    stance="supports" if delta > 0 else "opposes",
                    weight=min(0.20, abs(delta) * 1.5),
                    tags=("base_rate", "resolution"),
                )
            )
    return evidence


def structured_time_series_evidence(target: ForecastQuestion, question: dict[str, Any]) -> list[EvidenceItem]:
    source = str(question.get("source", ""))
    text = (str(question.get("question", "")) + " " + str(question.get("background", ""))).lower()
    bucket = str(target.metadata.get("horizon_bucket", "unknown"))
    resolution_month = _month(target.resolution_date)
    freeze_value = _float_or_none(question.get("freeze_datetime_value"))
    items: list[EvidenceItem] = []

    if source == "dbnomics" and "temperature" in text:
        if bucket == "short" and resolution_month in {1, 2}:
            weight = 0.18 if freeze_value is None or freeze_value >= 10.0 else 0.07
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "DBnomics temperature question resolves in winter; freeze-time temperature already above a low baseline, reducing odds of a higher later value.",
                    "opposes",
                    weight,
                    "recent",
                )
            )
        elif bucket == "mid" and resolution_month in {4, 5, 6, 7} and (freeze_value is None or freeze_value < 20.0):
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "DBnomics temperature question resolves in spring/summer from a non-hot freeze-time value, increasing odds of a higher later value.",
                    "supports",
                    0.16,
                    "recent",
                )
            )
    elif source == "yfinance" and ("market close price" in text or "stock" in text):
        if bucket == "near":
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "Near-horizon equity price direction has high noise; avoid treating long-run equity drift as immediate support.",
                    "opposes",
                    0.06,
                    "counterevidence",
                )
            )
        elif bucket in {"short", "mid"}:
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "Short-to-mid-horizon equity price questions have modest upward drift absent stronger contradictory evidence.",
                    "supports",
                    0.08,
                    "recent",
                )
            )
    elif source == "fred":
        if bucket == "mid":
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "FRED increase questions at mid horizon have enough time for noisy macro series to move above the freeze-time value.",
                    "supports",
                    0.08,
                    "recent",
                )
            )
        if "yield" in text and freeze_value is not None and freeze_value >= 6.0:
            items.append(
                _item(
                    target.id,
                    "structured_timeseries",
                    "High freeze-time yield values identify volatile rate/spread series where further increases remain plausible.",
                    "supports",
                    0.06,
                    "recent",
                )
            )
    return items


def _item(target_id: str, source: str, text: str, stance: str, weight: float, tag: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"{target_id}-{source}",
        source=source,
        text=text,
        stance=stance,
        weight=weight,
        tags=(tag,),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _month(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).month
    except ValueError:
        return None


def render_text(text: str, forecast_due_date: str, resolution_date: str | None) -> str:
    return (
        str(text)
        .replace("{forecast_due_date}", forecast_due_date or "")
        .replace("{resolution_date}", resolution_date or "")
    )


def target_key(base_id: str, resolution_date: str | None) -> str:
    return f"{base_id}::{resolution_date or 'NA'}"


def load_forecaster_medians(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{path} is a Git LFS pointer, not the forecast JSON payload")
    data = json.loads(text)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in data.get("forecasts", []):
        base_id = row.get("id")
        if not isinstance(base_id, str):
            continue
        forecast = row.get("forecast")
        if not isinstance(forecast, (int, float)):
            continue
        key = target_key(base_id, row.get("resolution_date"))
        grouped[key].append(float(forecast))
    return {key: statistics.median(values) for key, values in grouped.items() if values}


def load_superforecaster_medians(path: Path) -> dict[str, float]:
    return load_forecaster_medians(path)
