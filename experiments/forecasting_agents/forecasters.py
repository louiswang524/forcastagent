from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from .models import AgentForecast, EvidenceItem, ForecastQuestion, ForecastResult, SubQuestion, clamp_probability


def logit(p: float) -> float:
    p = clamp_probability(p)
    return math.log(p / (1.0 - p))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def weighted_probability(prior: float, evidence: list[EvidenceItem], multiplier: float = 1.0) -> float:
    score = logit(prior) + sum(item.signed_weight() for item in evidence) * multiplier
    return clamp_probability(logistic(score))


def base_prior(question: ForecastQuestion, default: float = 0.5) -> float:
    value = question.metadata.get("base_prior", default)
    try:
        return clamp_probability(float(value))
    except (TypeError, ValueError):
        return default


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.5
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def source_prior(question: ForecastQuestion, priors: dict[str, float]) -> float | None:
    source = str(question.metadata.get("forecastbench_source", question.domain))
    bucket = question.metadata.get("horizon_bucket")
    if bucket:
        prior = priors.get(f"{source}::{bucket}")
        if prior is not None:
            return prior
    return priors.get(source)


class Forecaster(ABC):
    name: str

    @abstractmethod
    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        raise NotImplementedError


class MarketOnlyForecaster(Forecaster):
    name = "market_only"

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        probability = question.market_probability if question.market_probability is not None else 0.5
        liquidity = float(question.metadata.get("market_liquidity", 0.5))
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=probability,
            confidence=max(0.05, min(0.95, liquidity)),
            rationale="Uses only the supplied market or crowd probability.",
            evidence=tuple(evidence),
            diagnostics={"market_liquidity": liquidity},
        )


class AIABaselineForecaster(Forecaster):
    """A compact offline analogue of multi-agent search, supervisor aggregation, and calibration."""

    name = "aia_baseline"

    def __init__(self, agent_count: int = 5, calibration_slope: float = 1.15) -> None:
        self.agent_count = agent_count
        self.calibration_slope = calibration_slope

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        prior = question.market_probability if question.market_probability is not None else 0.5
        groups = [
            [item for item in evidence if "base_rate" in item.tags],
            [item for item in evidence if "recent" in item.tags],
            [item for item in evidence if "counterevidence" in item.tags],
            [item for item in evidence if "market" in item.tags],
            evidence,
        ]
        agents: list[AgentForecast] = []
        for idx in range(self.agent_count):
            selected = groups[idx % len(groups)] or evidence
            probability = weighted_probability(prior, selected, multiplier=0.95)
            confidence = min(0.95, 0.45 + sum(abs(item.weight) for item in selected))
            agents.append(
                AgentForecast(
                    agent=f"search_agent_{idx + 1}",
                    probability=probability,
                    confidence=confidence,
                    rationale=f"Forecast from {len(selected)} focused evidence items.",
                    evidence_ids=tuple(item.id for item in selected),
                )
            )
        supervisor_probability = median([agent.probability for agent in agents])
        disagreement = max(agent.probability for agent in agents) - min(agent.probability for agent in agents)
        calibrated = logistic(logit(supervisor_probability) * self.calibration_slope)
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=calibrated,
            confidence=max(0.05, 0.85 - disagreement),
            rationale="Median ensemble with supervisor-style disagreement penalty and fixed extremization.",
            component_forecasts=tuple(agents),
            evidence=tuple(evidence),
            diagnostics={"disagreement": disagreement, "uncalibrated_probability": supervisor_probability},
        )


class EvidenceGraphForecaster(Forecaster):
    name = "evidence_graph_v1"

    def __init__(self, use_market: bool = True) -> None:
        self.use_market = use_market

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        graph = EvidenceGraph(evidence)
        prior = base_prior(question)
        subquestions = decompose_question(question, prior)
        agents = [
            specialist_forecast("base_rate_agent", question, graph.by_tag("base_rate"), prior=subquestions[0].prior),
            specialist_forecast("recent_evidence_agent", question, graph.by_tag("recent"), prior=prior),
            specialist_forecast("counterevidence_agent", question, graph.by_tag("counterevidence"), prior=prior),
            specialist_forecast("resolution_agent", question, graph.by_tag("resolution") or evidence, prior=prior),
        ]
        market_probability = question.market_probability if self.use_market else None
        probability = aggregate_subforecasts(subquestions, agents, market_probability)
        contradiction = graph.contradiction_score()
        confidence = max(0.05, min(0.95, 0.75 - contradiction * 0.25 + len(evidence) * 0.02))
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=probability,
            confidence=confidence,
            rationale="Evidence graph groups claims by role, then aggregates specialist forecasts against decomposed sub-questions.",
            component_forecasts=tuple(agents),
            evidence=tuple(evidence),
            subquestions=tuple(subquestions),
            diagnostics={"contradiction_score": contradiction, "tag_counts": dict(graph.tag_counts)},
        )


class CalibratedSourcePriorForecaster(EvidenceGraphForecaster):
    name = "calibrated_source_prior"

    def __init__(
        self,
        source_priors: dict[str, float],
        use_market: bool = True,
    ) -> None:
        super().__init__(use_market=use_market)
        self.source_priors = source_priors

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        prior = source_prior(question, self.source_priors)
        if prior is None:
            return super().forecast(question, evidence)
        calibrated_question = ForecastQuestion(
            **{**question.__dict__, "metadata": {**question.metadata, "base_prior": prior}}
        )
        result = super().forecast(calibrated_question, evidence)
        return ForecastResult(
            question_id=result.question_id,
            system=self.name,
            probability=result.probability,
            confidence=result.confidence,
            rationale="Evidence graph with source priors learned from calibration targets.",
            component_forecasts=result.component_forecasts,
            evidence=result.evidence,
            subquestions=result.subquestions,
            diagnostics={**result.diagnostics, "calibrated_source_prior": prior},
        )


class EvidenceGraphNoMarketForecaster(EvidenceGraphForecaster):
    name = "evidence_graph_no_market"

    def __init__(self) -> None:
        super().__init__(use_market=False)


class StructuredEvidenceGraphForecaster(EvidenceGraphForecaster):
    name = "structured_evidence_graph"

    def __init__(self, structured_multiplier: float = 1.75) -> None:
        super().__init__(use_market=True)
        self.structured_multiplier = structured_multiplier

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        amplified = [
            EvidenceItem(
                id=item.id,
                source=item.source,
                text=item.text,
                stance=item.stance,
                weight=min(0.35, item.weight * self.structured_multiplier) if item.source == "structured_timeseries" else item.weight,
                date=item.date,
                tags=item.tags,
            )
            for item in evidence
        ]
        result = super().forecast(question, amplified)
        return ForecastResult(
            question_id=result.question_id,
            system=self.name,
            probability=result.probability,
            confidence=result.confidence,
            rationale="Evidence graph with amplified structured time-series evidence.",
            component_forecasts=result.component_forecasts,
            evidence=result.evidence,
            subquestions=result.subquestions,
            diagnostics={**result.diagnostics, "structured_multiplier": self.structured_multiplier},
        )


class AdvancedForecastLab(Forecaster):
    name = "advanced_v1"

    def __init__(
        self,
        domain_calibration: dict[str, float] | None = None,
        use_market: bool = True,
        use_error_memory: bool = True,
        use_domain_calibration: bool = True,
    ) -> None:
        self.use_market = use_market
        self.use_error_memory = use_error_memory
        self.use_domain_calibration = use_domain_calibration
        self.domain_calibration = domain_calibration or {
            "politics": 1.12,
            "business": 0.92,
            "regulation": 0.88,
            "default": 1.0,
        }
        self.error_memory = [
            "Rumor-heavy acquisition questions are often over-forecast without hard financing or board evidence.",
            "Regulatory questions with litigation or objections often miss deadline even when technical review is favorable.",
            "Political ballot questions with stable polling and campaign funding advantages can justify moving above market.",
        ]

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        graph = EvidenceGraph(evidence)
        blind = EvidenceGraphForecaster().forecast(
            ForecastQuestion(**{**question.__dict__, "market_probability": None}),
            evidence,
        )
        market_prior = question.market_probability if self.use_market and question.market_probability is not None else 0.5
        market_aware_question = question if self.use_market else ForecastQuestion(**{**question.__dict__, "market_probability": None})
        market_aware = EvidenceGraphForecaster().forecast(market_aware_question, evidence)
        adversarial = specialist_forecast("adversarial_reviewer", question, graph.opposing(), prior=market_prior)
        scenario = scenario_simulation_agent(question, evidence)
        liquidity = float(question.metadata.get("market_liquidity", 0.5))
        market_weight = min(0.65, max(0.20, liquidity)) if self.use_market else 0.0
        ai_weight = 1.0 - market_weight
        mixed_ai = 0.35 * blind.probability + 0.40 * market_aware.probability + 0.15 * scenario.probability + 0.10 * adversarial.probability
        warning_count = 0
        if self.use_error_memory:
            warning_count = error_memory_hits(question)
        memory_penalty = warning_count * 0.15
        mixed = logistic(logit(market_weight * market_prior + ai_weight * mixed_ai) - memory_penalty)
        slope = self.domain_calibration.get(question.domain, self.domain_calibration["default"]) if self.use_domain_calibration else 1.0
        calibrated = logistic(logit(mixed) * slope)
        confidence = max(0.05, min(0.95, 0.82 - graph.contradiction_score() * 0.2 - warning_count * 0.03))
        components = list(blind.component_forecasts) + list(market_aware.component_forecasts) + [adversarial, scenario]
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=calibrated,
            confidence=confidence,
            rationale="Dual blind/market tracks, adversarial review, scenario simulation, error memory, and domain calibration.",
            component_forecasts=tuple(components),
            evidence=tuple(evidence),
            subquestions=blind.subquestions,
            diagnostics={
                "blind_probability": blind.probability,
                "market_aware_probability": market_aware.probability,
                "market_weight": market_weight,
                "mixed_uncalibrated": mixed,
                "error_memory_hits": warning_count,
            },
        )


class AdvancedNoMarketForecastLab(AdvancedForecastLab):
    name = "advanced_no_market"

    def __init__(self) -> None:
        super().__init__(use_market=False)


class AdvancedNoMemoryForecastLab(AdvancedForecastLab):
    name = "advanced_no_memory"

    def __init__(self) -> None:
        super().__init__(use_error_memory=False)


class AdvancedNoCalibrationForecastLab(AdvancedForecastLab):
    name = "advanced_no_calibration"

    def __init__(self) -> None:
        super().__init__(use_domain_calibration=False)


class EvidenceGraph:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence
        self.tag_counts = Counter(tag for item in evidence for tag in item.tags)
        self._by_tag: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in evidence:
            for tag in item.tags:
                self._by_tag[tag].append(item)

    def by_tag(self, tag: str) -> list[EvidenceItem]:
        return list(self._by_tag.get(tag, []))

    def opposing(self) -> list[EvidenceItem]:
        return [item for item in self.evidence if item.stance == "opposes"]

    def contradiction_score(self) -> float:
        support = sum(item.weight for item in self.evidence if item.stance == "supports")
        oppose = sum(item.weight for item in self.evidence if item.stance == "opposes")
        total = support + oppose
        if total <= 0:
            return 0.0
        return min(support, oppose) / total


def error_memory_hits(question: ForecastQuestion) -> int:
    title = question.title.lower()
    domain = question.domain.lower()
    hits = 0
    if domain == "business" and any(term in title for term in ["acquisition", "acquire", "merger", "takeover"]):
        hits += 1
    if domain == "regulation" and any(term in title for term in ["permit", "approve", "regulator", "approval"]):
        hits += 1
    if domain == "politics" and any(term in title for term in ["ballot", "election", "referendum"]):
        hits += 1
    return hits


def decompose_question(question: ForecastQuestion, prior: float | None = None) -> list[SubQuestion]:
    prior = base_prior(question) if prior is None else prior
    return [
        SubQuestion("base_rate", f"What is the historical base rate for {question.domain} questions like this?", 0.35, prior),
        SubQuestion("recent_signal", "Does recent evidence materially change the base rate?", 0.25, prior),
        SubQuestion("deadline", f"Can the event satisfy the resolution criteria by {question.resolution_date}?", 0.25, prior),
        SubQuestion("counterevidence", "What evidence would make the event fail?", 0.15, prior),
    ]


def specialist_forecast(name: str, question: ForecastQuestion, evidence: list[EvidenceItem], prior: float) -> AgentForecast:
    if not evidence:
        probability = prior
        confidence = 0.25
    else:
        probability = weighted_probability(prior, evidence)
        confidence = min(0.95, 0.35 + sum(abs(item.weight) for item in evidence))
    return AgentForecast(
        agent=name,
        probability=probability,
        confidence=confidence,
        rationale=f"{name} used {len(evidence)} evidence items for {question.domain}.",
        evidence_ids=tuple(item.id for item in evidence),
    )


def aggregate_subforecasts(
    subquestions: list[SubQuestion],
    agents: list[AgentForecast],
    market_probability: float | None,
    market_weight: float = 0.25,
) -> float:
    by_name = {agent.agent: agent for agent in agents}
    mapping = {
        "base_rate": by_name.get("base_rate_agent"),
        "recent_signal": by_name.get("recent_evidence_agent"),
        "deadline": by_name.get("resolution_agent"),
        "counterevidence": by_name.get("counterevidence_agent"),
    }
    score = 0.0
    total = 0.0
    for subquestion in subquestions:
        agent = mapping.get(subquestion.id)
        if agent is None:
            continue
        score += logit(agent.probability) * subquestion.weight * max(0.25, agent.confidence)
        total += subquestion.weight * max(0.25, agent.confidence)
    if market_probability is not None:
        score += logit(market_probability) * market_weight
        total += market_weight
    if total == 0:
        return 0.5
    return clamp_probability(logistic(score / total))


class WeightedEvidenceGraphForecaster(Forecaster):
    name = "weighted_evidence_graph"

    def __init__(
        self,
        source_priors: dict[str, float] | None = None,
        subquestion_weights: dict[str, float] | None = None,
        market_weight: float = 0.25,
        evidence_multiplier: float = 1.0,
    ) -> None:
        self.source_priors = source_priors or {}
        self.subquestion_weights = subquestion_weights or {
            "base_rate": 0.35,
            "recent_signal": 0.25,
            "deadline": 0.25,
            "counterevidence": 0.15,
        }
        self.market_weight = market_weight
        self.evidence_multiplier = evidence_multiplier

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        prior = source_prior(question, self.source_priors)
        if prior is None:
            prior = base_prior(question)
        calibrated_question = ForecastQuestion(
            **{**question.__dict__, "metadata": {**question.metadata, "base_prior": prior}}
        )
        graph = EvidenceGraph(evidence)
        subquestions = [
            SubQuestion("base_rate", f"What is the historical base rate for {question.domain} questions like this?", self.subquestion_weights.get("base_rate", 0.0), prior),
            SubQuestion("recent_signal", "Does recent evidence materially change the base rate?", self.subquestion_weights.get("recent_signal", 0.0), prior),
            SubQuestion("deadline", f"Can the event satisfy the resolution criteria by {question.resolution_date}?", self.subquestion_weights.get("deadline", 0.0), prior),
            SubQuestion("counterevidence", "What evidence would make the event fail?", self.subquestion_weights.get("counterevidence", 0.0), prior),
        ]
        agents = [
            specialist_forecast("base_rate_agent", calibrated_question, graph.by_tag("base_rate"), prior=prior),
            specialist_forecast("recent_evidence_agent", calibrated_question, graph.by_tag("recent"), prior=prior),
            specialist_forecast("counterevidence_agent", calibrated_question, graph.by_tag("counterevidence"), prior=prior),
            specialist_forecast("resolution_agent", calibrated_question, graph.by_tag("resolution") or evidence, prior=prior),
        ]
        if self.evidence_multiplier != 1.0:
            agents = [
                AgentForecast(
                    agent=agent.agent,
                    probability=logistic(logit(prior) + (logit(agent.probability) - logit(prior)) * self.evidence_multiplier),
                    confidence=agent.confidence,
                    rationale=agent.rationale,
                    evidence_ids=agent.evidence_ids,
                )
                for agent in agents
            ]
        probability = aggregate_subforecasts(subquestions, agents, question.market_probability, self.market_weight)
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=probability,
            confidence=0.75,
            rationale="Weighted evidence graph candidate from self-improvement sweep.",
            component_forecasts=tuple(agents),
            evidence=tuple(evidence),
            subquestions=tuple(subquestions),
            diagnostics={
                "source_prior": prior,
                "market_weight": self.market_weight,
                "evidence_multiplier": self.evidence_multiplier,
                "subquestion_weights": self.subquestion_weights,
            },
        )


def scenario_simulation_agent(question: ForecastQuestion, evidence: list[EvidenceItem]) -> AgentForecast:
    support = sum(item.weight for item in evidence if item.stance == "supports")
    oppose = sum(item.weight for item in evidence if item.stance == "opposes")
    net = support - oppose
    deadline_penalty = 0.04 if any(word in question.resolution_criteria.lower() for word in ["before", "by", "deadline"]) else 0.0
    probability = clamp_probability(logistic(net - deadline_penalty))
    return AgentForecast(
        agent="scenario_simulation_agent",
        probability=probability,
        confidence=min(0.90, 0.40 + abs(net)),
        rationale="Approximates the share of plausible scenarios where supporting evidence beats blockers by the deadline.",
        evidence_ids=tuple(item.id for item in evidence),
    )



