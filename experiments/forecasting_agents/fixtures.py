from __future__ import annotations

from .models import EvidenceItem, ForecastQuestion


SAMPLE_QUESTIONS = [
    ForecastQuestion(
        id="q1",
        title="Will the city approve the downtown transit bond by election day?",
        description="A local ballot measure requires a simple majority to pass.",
        resolution_criteria="Resolve true if official election results certify passage by the deadline.",
        domain="politics",
        cutoff_date="2026-06-01",
        resolution_date="2026-11-10",
        outcome=1,
        market_probability=0.58,
    ),
    ForecastQuestion(
        id="q2",
        title="Will Acme Robotics announce an acquisition of BetaMotion before Q4?",
        description="Rumors suggest a possible acquisition, but no formal process has been announced.",
        resolution_criteria="Resolve true if Acme Robotics or BetaMotion announces a definitive acquisition agreement before 2026-10-01.",
        domain="business",
        cutoff_date="2026-06-01",
        resolution_date="2026-10-01",
        outcome=0,
        market_probability=0.41,
    ),
    ForecastQuestion(
        id="q3",
        title="Will the energy regulator approve the offshore wind permit by year end?",
        description="The regulator is reviewing a large offshore wind permit with environmental objections pending.",
        resolution_criteria="Resolve true if the regulator publishes final approval before 2026-12-31.",
        domain="regulation",
        cutoff_date="2026-06-01",
        resolution_date="2026-12-31",
        outcome=0,
        market_probability=0.35,
    ),
]


SAMPLE_EVIDENCE = {
    "q1": [
        EvidenceItem("q1-e1", "poll", "Three recent polls show support above 55 percent.", "supports", 0.22, tags=("recent", "poll")),
        EvidenceItem("q1-e2", "finance", "Opposition campaign has raised less than half of supporters.", "supports", 0.08, tags=("campaign",)),
        EvidenceItem("q1-e3", "history", "Similar bonds passed in four of the last five local elections.", "supports", 0.12, tags=("base_rate",)),
        EvidenceItem("q1-e4", "news", "A taxpayer group launched a late opposition campaign.", "opposes", 0.09, tags=("counterevidence",)),
    ],
    "q2": [
        EvidenceItem("q2-e1", "rumor", "Industry newsletter reports early acquisition talks.", "supports", 0.12, tags=("recent",)),
        EvidenceItem("q2-e2", "filing", "Acme CFO said capital allocation will prioritize buybacks.", "opposes", 0.20, tags=("counterevidence",)),
        EvidenceItem("q2-e3", "history", "Acme has completed only one acquisition in the last decade.", "opposes", 0.15, tags=("base_rate",)),
        EvidenceItem("q2-e4", "market", "BetaMotion shares trade at only a small takeover premium.", "opposes", 0.08, tags=("market",)),
    ],
    "q3": [
        EvidenceItem("q3-e1", "agency", "Agency staff issued a favorable technical review.", "supports", 0.13, tags=("recent",)),
        EvidenceItem("q3-e2", "court", "Pending litigation can delay final approval beyond deadline.", "opposes", 0.18, tags=("counterevidence",)),
        EvidenceItem("q3-e3", "history", "Comparable permits usually take more than twelve months after objections.", "opposes", 0.16, tags=("base_rate",)),
        EvidenceItem("q3-e4", "politics", "Governor publicly supports the project.", "supports", 0.07, tags=("politics",)),
    ],
}

