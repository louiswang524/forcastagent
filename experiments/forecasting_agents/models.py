from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def clamp_probability(value: float) -> float:
    return max(0.01, min(0.99, value))


@dataclass(frozen=True)
class ForecastQuestion:
    id: str
    title: str
    description: str
    resolution_criteria: str
    domain: str
    cutoff_date: str
    resolution_date: str
    outcome: int | None = None
    market_probability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    source: str
    text: str
    stance: str
    weight: float
    date: str | None = None
    tags: tuple[str, ...] = ()

    def signed_weight(self) -> float:
        if self.stance == "supports":
            return self.weight
        if self.stance == "opposes":
            return -self.weight
        return 0.0


@dataclass(frozen=True)
class SubQuestion:
    id: str
    text: str
    weight: float
    prior: float


@dataclass(frozen=True)
class AgentForecast:
    agent: str
    probability: float
    confidence: float
    rationale: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastResult:
    question_id: str
    system: str
    probability: float
    confidence: float
    rationale: str
    component_forecasts: tuple[AgentForecast, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    subquestions: tuple[SubQuestion, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def scored_probability(self) -> float:
        return clamp_probability(self.probability)

