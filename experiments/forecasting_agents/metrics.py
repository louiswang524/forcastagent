from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .models import ForecastResult


def brier_score(probability: float, outcome: int) -> float:
    return (probability - float(outcome)) ** 2


def log_score(probability: float, outcome: int) -> float:
    p = max(0.001, min(0.999, probability))
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def calibration_error(rows: Iterable[tuple[float, int]], buckets: int = 10) -> float:
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for probability, outcome in rows:
        bucket = min(buckets - 1, int(probability * buckets))
        grouped[bucket].append((probability, outcome))
    if not grouped:
        return 0.0
    total = 0
    weighted = 0.0
    for items in grouped.values():
        avg_p = sum(p for p, _ in items) / len(items)
        avg_y = sum(y for _, y in items) / len(items)
        weighted += abs(avg_p - avg_y) * len(items)
        total += len(items)
    return weighted / total


def summarize_results(results: list[ForecastResult], outcomes: dict[str, int]) -> dict[str, dict[str, float]]:
    by_system: dict[str, list[ForecastResult]] = defaultdict(list)
    for result in results:
        by_system[result.system].append(result)
    summary: dict[str, dict[str, float]] = {}
    for system, system_results in by_system.items():
        scored = [(r.scored_probability(), outcomes[r.question_id]) for r in system_results if r.question_id in outcomes]
        if not scored:
            continue
        summary[system] = {
            "n": float(len(scored)),
            "brier": sum(brier_score(p, y) for p, y in scored) / len(scored),
            "log_score": sum(log_score(p, y) for p, y in scored) / len(scored),
            "calibration_error": calibration_error(scored),
        }
    return summary

