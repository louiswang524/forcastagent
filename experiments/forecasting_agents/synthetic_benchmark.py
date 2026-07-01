from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .forecasters import logistic
from .models import EvidenceItem, ForecastQuestion, clamp_probability


DOMAINS = ("politics", "business", "regulation")


def generate_synthetic_dataset(
    n: int,
    seed: int,
    out_path: Path | None = None,
) -> tuple[list[ForecastQuestion], dict[str, list[EvidenceItem]]]:
    rng = random.Random(seed)
    questions: list[ForecastQuestion] = []
    evidence_by_question: dict[str, list[EvidenceItem]] = {}
    for idx in range(n):
        domain = DOMAINS[idx % len(DOMAINS)]
        horizon_days = rng.choice([30, 60, 120, 240, 360])
        liquidity = rng.betavariate(2.2, 2.2)
        latent_logit = domain_bias(domain) + rng.gauss(0.0, 0.9)
        base_signal = latent_logit + rng.gauss(0.0, 0.55)
        recent_signal = latent_logit + rng.gauss(0.0, 0.75)
        resolution_signal = latent_logit - horizon_days / 900.0 + rng.gauss(0.0, 0.65)
        counter_signal = -latent_logit + rng.gauss(0.0, 0.65)
        true_probability = clamp_probability(logistic(latent_logit))
        outcome = int(rng.random() < true_probability)
        market_noise = rng.gauss(0.0, 0.55 + (1.0 - liquidity) * 0.95)
        market_probability = clamp_probability(logistic(latent_logit + market_noise))
        question_id = f"synthetic_{seed}_{idx:04d}"
        question = ForecastQuestion(
            id=question_id,
            title=synthetic_title(domain, idx),
            description=f"Synthetic {domain} event with hidden true probability {true_probability:.3f}.",
            resolution_criteria=f"Resolve true if the synthetic event occurs within {horizon_days} days.",
            domain=domain,
            cutoff_date="2026-01-01",
            resolution_date=f"2026-{min(12, 1 + horizon_days // 30):02d}-28",
            outcome=outcome,
            market_probability=market_probability,
            metadata={
                "market_liquidity": round(liquidity, 4),
                "horizon_days": horizon_days,
                "latent_probability": round(true_probability, 6),
                "generator_seed": seed,
            },
        )
        evidence = [
            evidence_item(question_id, "base_rate", base_signal, rng, tag="base_rate"),
            evidence_item(question_id, "recent", recent_signal, rng, tag="recent"),
            evidence_item(question_id, "resolution", resolution_signal, rng, tag="resolution"),
            evidence_item(question_id, "counter", counter_signal, rng, tag="counterevidence", invert=True),
        ]
        if rng.random() < 0.45:
            evidence.append(evidence_item(question_id, "market_context", latent_logit + market_noise, rng, tag="market"))
        questions.append(question)
        evidence_by_question[question_id] = evidence
    if out_path is not None:
        write_dataset(out_path, questions, evidence_by_question)
    return questions, evidence_by_question


def domain_bias(domain: str) -> float:
    if domain == "politics":
        return 0.15
    if domain == "business":
        return -0.20
    if domain == "regulation":
        return -0.10
    return 0.0


def synthetic_title(domain: str, idx: int) -> str:
    if domain == "business":
        return f"Will Company {idx} announce the synthetic acquisition before deadline?"
    if domain == "regulation":
        return f"Will regulator {idx} approve the synthetic permit by deadline?"
    return f"Will ballot measure {idx} pass in the synthetic election?"


def evidence_item(question_id: str, source: str, signal: float, rng: random.Random, tag: str, invert: bool = False) -> EvidenceItem:
    stance_signal = -signal if invert else signal
    stance = "supports" if stance_signal >= 0 else "opposes"
    strength = min(0.34, max(0.04, abs(signal) * 0.13 + rng.uniform(0.01, 0.06)))
    return EvidenceItem(
        id=f"{question_id}-{source}",
        source=source,
        text=f"Synthetic {source} signal has signed strength {signal:.3f}.",
        stance=stance,
        weight=round(strength, 4),
        tags=(tag,),
    )


def write_dataset(path: Path, questions: list[ForecastQuestion], evidence_by_question: dict[str, list[EvidenceItem]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "questions": [
            {
                "id": q.id,
                "title": q.title,
                "description": q.description,
                "resolution_criteria": q.resolution_criteria,
                "domain": q.domain,
                "cutoff_date": q.cutoff_date,
                "resolution_date": q.resolution_date,
                "outcome": q.outcome,
                "market_probability": q.market_probability,
                "metadata": q.metadata,
            }
            for q in questions
        ],
        "evidence": {
            qid: [
                {
                    "id": item.id,
                    "source": item.source,
                    "text": item.text,
                    "stance": item.stance,
                    "weight": item.weight,
                    "date": item.date,
                    "tags": list(item.tags),
                }
                for item in items
            ]
            for qid, items in evidence_by_question.items()
        },
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

