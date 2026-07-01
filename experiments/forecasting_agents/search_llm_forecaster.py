from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .forecasters import EvidenceGraphForecaster, base_prior, logistic, logit
from .models import AgentForecast, EvidenceItem, ForecastQuestion, ForecastResult, clamp_probability


@dataclass(frozen=True)
class SearchQuery:
    id: str
    text: str
    purpose: str


@dataclass(frozen=True)
class SearchDocument:
    id: str
    query_id: str
    title: str
    snippet: str
    url: str = ""
    date: str = ""
    source: str = "local"


@dataclass(frozen=True)
class ReasonedForecast:
    probability: float
    confidence: float
    rationale: str
    evidence_ids: tuple[str, ...]


class SearchProvider(Protocol):
    def search(self, question: ForecastQuestion, evidence: list[EvidenceItem], query: SearchQuery) -> list[SearchDocument]:
        ...


class Reasoner(Protocol):
    def forecast(self, question: ForecastQuestion, documents: list[SearchDocument], prior: float, role: str) -> ReasonedForecast:
        ...


class LocalEvidenceSearchProvider:
    """Offline search over harness evidence and optional historical analog probabilities."""

    def __init__(self, historical_probabilities: dict[str, float] | None = None) -> None:
        self.historical_probabilities = historical_probabilities or {}

    def search(self, question: ForecastQuestion, evidence: list[EvidenceItem], query: SearchQuery) -> list[SearchDocument]:
        docs: list[SearchDocument] = []
        for item in evidence:
            if query.purpose == "base_rate" and "base_rate" not in item.tags:
                continue
            if query.purpose == "recent" and "recent" not in item.tags and "market" not in item.tags:
                continue
            if query.purpose == "counterevidence" and item.stance != "opposes" and "counterevidence" not in item.tags:
                continue
            if query.purpose == "resolution" and "resolution" not in item.tags:
                continue
            docs.append(
                SearchDocument(
                    id=item.id,
                    query_id=query.id,
                    title=item.source,
                    snippet=f"{item.stance}: {item.text} [weight={item.weight:.3f}]",
                    date=item.date or str(question.metadata.get("forecast_due_date", "")),
                    source=item.source,
                )
            )
        if query.purpose in {"base_rate", "recent"} and question.id in self.historical_probabilities:
            probability = self.historical_probabilities[question.id]
            docs.append(
                SearchDocument(
                    id=f"{question.id}-historical-analog",
                    query_id=query.id,
                    title="historical_analog",
                    snippet=f"Leakage-checked historical analog probability for same-horizon increase: {probability:.4f}.",
                    date=str(question.metadata.get("forecast_due_date", "")),
                    source="historical_analog",
                )
            )
        if question.market_probability is not None and query.purpose == "recent":
            docs.append(
                SearchDocument(
                    id=f"{question.id}-market-probability",
                    query_id=query.id,
                    title="market_or_crowd_probability",
                    snippet=f"Freeze-time market/crowd probability: {question.market_probability:.4f}.",
                    date=str(question.metadata.get("forecast_due_date", "")),
                    source="market",
                )
            )
        return docs


class HeuristicReasoner:
    """Deterministic stand-in for an LLM judge, used for reproducible tests and offline runs."""

    def forecast(self, question: ForecastQuestion, documents: list[SearchDocument], prior: float, role: str) -> ReasonedForecast:
        score = logit(prior)
        evidence_ids: list[str] = []
        for doc in documents:
            snippet = doc.snippet.lower()
            evidence_ids.append(doc.id)
            analog = _extract_labeled_float(snippet, "probability")
            market = _extract_labeled_float(snippet, "market/crowd probability")
            if analog is not None:
                score += (logit(analog) - logit(prior)) * 0.78
                continue
            if market is not None:
                score += (logit(market) - logit(prior)) * 0.55
                continue
            weight = _extract_labeled_float(snippet, "weight") or 0.04
            if "supports:" in snippet:
                score += min(0.7, weight * 1.35)
            elif "opposes:" in snippet:
                score -= min(0.7, weight * 1.35)
            elif "neutral:" in snippet:
                score += 0.0
        probability = clamp_probability(logistic(score))
        confidence = min(0.90, 0.36 + 0.05 * len(documents))
        return ReasonedForecast(
            probability=probability,
            confidence=confidence,
            rationale=f"{role} used {len(documents)} retrieved documents with prior {prior:.3f}.",
            evidence_ids=tuple(evidence_ids),
        )


class OpenAICompatibleReasoner:
    """Optional live LLM reasoner using an OpenAI-compatible chat-completions endpoint."""

    SYSTEM_PROMPTS = {
        "plain": "You are a calibrated binary event forecaster. Return strict JSON with probability, confidence, rationale.",
        "checklist": (
            "You are a calibrated binary event forecaster. Estimate the probability that the question resolves true. "
            "Reason from base rates, recent evidence, resolution criteria, and counterevidence. Use the prior as an anchor, "
            "but update away from it when the retrieved documents justify it. Avoid vague 0.5 hedging. Return strict JSON "
            "with numeric probability, numeric confidence, and concise rationale."
        ),
        "extremized": (
            "You are a calibrated superforecasting-style binary event forecaster. Estimate the probability that the question "
            "resolves true. Start from the prior, separate base-rate evidence from current-state evidence, and then make the "
            "forecast sharp enough to reflect the evidence strength. LLMs tend to understate probabilities away from 0.5; "
            "correct for that only when evidence is coherent. Return strict JSON with numeric probability, numeric confidence, "
            "and concise rationale."
        ),
        "conservative": (
            "You are a cautious calibrated binary event forecaster. Estimate the probability that the question resolves true. "
            "Put strong weight on base rates and resolution criteria, discount weak textual cues, and move far from the prior "
            "only when dated evidence is directly relevant. Return strict JSON with numeric probability, numeric confidence, "
            "and concise rationale."
        ),
    }

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        prompt_style: str = "plain",
        temperature: float = 0.0,
    ) -> None:
        if prompt_style not in self.SYSTEM_PROMPTS:
            choices = ", ".join(sorted(self.SYSTEM_PROMPTS))
            raise ValueError(f"Unknown prompt_style {prompt_style!r}; expected one of: {choices}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.prompt_style = prompt_style
        self.temperature = temperature

    def forecast(self, question: ForecastQuestion, documents: list[SearchDocument], prior: float, role: str) -> ReasonedForecast:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is required for OpenAICompatibleReasoner")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPTS[self.prompt_style],
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "role": role,
                            "question": question.title,
                            "resolution_criteria": question.resolution_criteria,
                            "forecast_due_date": question.metadata.get("forecast_due_date", question.cutoff_date),
                            "resolution_date": question.resolution_date,
                            "prior": prior,
                            "documents": [doc.__dict__ for doc in documents],
                            "instruction": (
                                "Estimate P(question resolves true). Use only documents dated no later than forecast_due_date. "
                                "Return JSON exactly like {\"probability\": 0.37, \"confidence\": 0.62, \"rationale\": \"...\"}."
                            ),
                        },
                        indent=2,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_reasoner_response(content)
        probability = clamp_probability(float(parsed["probability"]))
        confidence = max(0.05, min(0.95, float(parsed.get("confidence", 0.55))))
        return ReasonedForecast(
            probability=probability,
            confidence=confidence,
            rationale=str(parsed.get("rationale", "")),
            evidence_ids=tuple(doc.id for doc in documents),
        )


class SearchEnabledLLMForecaster:
    name = "search_llm_loop"

    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        reasoner: Reasoner | None = None,
        evidence_weight: float = 1.0,
        analog_gate: bool = True,
    ) -> None:
        self.search_provider = search_provider or LocalEvidenceSearchProvider()
        self.reasoner = reasoner or HeuristicReasoner()
        self.evidence_weight = evidence_weight
        self.analog_gate = analog_gate
        self.graph = EvidenceGraphForecaster()

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        prior = base_prior(question)
        queries = plan_search_queries(question)
        agent_forecasts: list[AgentForecast] = []
        documents: list[SearchDocument] = []
        for query in queries:
            query_docs = self.search_provider.search(question, evidence, query)
            documents.extend(query_docs)
            reasoned = self.reasoner.forecast(question, query_docs, prior, query.purpose)
            agent_forecasts.append(
                AgentForecast(
                    agent=f"llm_{query.purpose}_agent",
                    probability=reasoned.probability,
                    confidence=reasoned.confidence,
                    rationale=reasoned.rationale,
                    evidence_ids=reasoned.evidence_ids,
                )
            )
        graph_result = self.graph.forecast(question, evidence)
        agent_probability = aggregate_reasoned_forecasts(agent_forecasts, prior)
        analog_probability = retrieved_historical_probability(documents)
        analog_gate_fired = self.analog_gate and analog_probability is not None
        if analog_gate_fired:
            llm_probability = analog_probability
            probability = analog_probability
        elif analog_probability is None:
            llm_probability = agent_probability
            probability = self.evidence_weight * llm_probability + (1.0 - self.evidence_weight) * graph_result.probability
        else:
            llm_probability = 0.72 * analog_probability + 0.28 * agent_probability
            probability = self.evidence_weight * llm_probability + (1.0 - self.evidence_weight) * graph_result.probability
        disagreement = abs(llm_probability - graph_result.probability)
        return ForecastResult(
            question_id=question.id,
            system=self.name,
            probability=clamp_probability(probability),
            confidence=max(0.05, min(0.95, 0.80 - disagreement * 0.45)),
            rationale="Search-enabled loop with analog gate: use structured historical analogs when available, otherwise fall back to specialist search reasoning and evidence graph blending.",
            component_forecasts=tuple(agent_forecasts),
            evidence=tuple(
                EvidenceItem(
                    id=doc.id,
                    source=doc.source,
                    text=doc.snippet,
                    stance="neutral",
                    weight=0.0,
                    date=doc.date,
                    tags=("retrieved", doc.query_id),
                )
                for doc in documents
            ),
            diagnostics={
                "query_count": len(queries),
                "document_count": len(documents),
                "agent_probability": agent_probability,
                "historical_analog_probability": analog_probability,
                "analog_gate_enabled": self.analog_gate,
                "analog_gate_fired": analog_gate_fired,
                "llm_probability": llm_probability,
                "graph_probability": graph_result.probability,
                "disagreement": disagreement,
            },
        )


class HybridAnalogSearchForecaster:
    name = "hybrid_analog_search"

    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        reasoner: Reasoner | None = None,
        evidence_weight: float = 1.0,
    ) -> None:
        self.search_provider = search_provider or LocalEvidenceSearchProvider()
        self.search_forecaster = SearchEnabledLLMForecaster(
            search_provider=self.search_provider,
            reasoner=reasoner or HeuristicReasoner(),
            evidence_weight=evidence_weight,
            analog_gate=False,
        )

    def forecast(self, question: ForecastQuestion, evidence: list[EvidenceItem]) -> ForecastResult:
        prior = base_prior(question)
        analog_docs: list[SearchDocument] = []
        for query in plan_search_queries(question):
            if query.purpose not in {"base_rate", "recent"}:
                continue
            analog_docs.extend(self.search_provider.search(question, evidence, query))
        analog_probability = retrieved_historical_probability(analog_docs)
        if analog_probability is not None:
            return ForecastResult(
                question_id=question.id,
                system=self.name,
                probability=analog_probability,
                confidence=0.72,
                rationale="Hybrid gate used pure leakage-checked historical analog probability; search fallback was not needed.",
                evidence=tuple(
                    EvidenceItem(
                        id=doc.id,
                        source=doc.source,
                        text=doc.snippet,
                        stance="neutral",
                        weight=0.0,
                        date=doc.date,
                        tags=("retrieved", doc.query_id),
                    )
                    for doc in analog_docs
                ),
                diagnostics={
                    "gate_decision": "historical_analog",
                    "historical_analog_probability": analog_probability,
                    "fallback_probability": None,
                    "base_prior": prior,
                },
            )
        fallback = self.search_forecaster.forecast(question, evidence)
        return ForecastResult(
            question_id=fallback.question_id,
            system=self.name,
            probability=fallback.probability,
            confidence=fallback.confidence,
            rationale="Hybrid gate found no structured historical analog and used search/graph fallback.",
            component_forecasts=fallback.component_forecasts,
            evidence=fallback.evidence,
            subquestions=fallback.subquestions,
            diagnostics={
                **fallback.diagnostics,
                "gate_decision": "search_fallback",
                "fallback_probability": fallback.probability,
                "base_prior": prior,
            },
        )


def plan_search_queries(question: ForecastQuestion) -> list[SearchQuery]:
    source = question.metadata.get("forecastbench_source", question.domain)
    return [
        SearchQuery("base", f"base rate for {source} question like: {question.title}", "base_rate"),
        SearchQuery("recent", f"freeze-time evidence and current value for: {question.title}", "recent"),
        SearchQuery("resolution", f"resolution criteria and deadline risk for: {question.title}", "resolution"),
        SearchQuery("counter", f"counterevidence or blockers for: {question.title}", "counterevidence"),
    ]


def aggregate_reasoned_forecasts(forecasts: list[AgentForecast], prior: float) -> float:
    if not forecasts:
        return prior
    total = 0.0
    score = 0.0
    for forecast in forecasts:
        weight = max(0.20, forecast.confidence)
        score += logit(forecast.probability) * weight
        total += weight
    return clamp_probability(logistic(score / total))


def retrieved_historical_probability(documents: list[SearchDocument]) -> float | None:
    probabilities = []
    seen = set()
    for doc in documents:
        if doc.source != "historical_analog" or doc.id in seen:
            continue
        seen.add(doc.id)
        probability = _extract_labeled_float(doc.snippet.lower(), "probability")
        if probability is not None:
            probabilities.append(probability)
    if not probabilities:
        return None
    return clamp_probability(sum(probabilities) / len(probabilities))


def _extract_labeled_float(text: str, label: str) -> float | None:
    idx = text.find(label)
    if idx < 0:
        return None
    tail = text[idx + len(label) : idx + len(label) + 40]
    for sep in [":", "="]:
        if sep in tail:
            tail = tail.split(sep, 1)[1]
            break
    token = ""
    decimal_seen = False
    for char in tail.strip():
        if char.isdigit() or char == "-":
            token += char
        elif char == "." and not decimal_seen:
            token += char
            decimal_seen = True
        elif token:
            break
    try:
        return clamp_probability(float(token))
    except ValueError:
        return None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _parse_reasoner_response(text: str) -> dict[str, Any]:
    try:
        return _parse_json_object(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        probability = _extract_labeled_float(lowered, "probability")
        confidence = _extract_labeled_float(lowered, "confidence")
        if probability is None:
            probability = _extract_labeled_float(lowered, "p(")
        if probability is None:
            raise
        return {
            "probability": probability,
            "confidence": 0.55 if confidence is None else confidence,
            "rationale": "Recovered from malformed JSON response.",
        }
