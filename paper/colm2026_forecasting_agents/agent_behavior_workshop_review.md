# Academic Research Paper Review: WAB/COLM 2026 Fit

Paper reviewed: "When Should Forecasting Agents Reason? Behavioral Reliability Routing for ForecastBench"

Venue target: Workshop on Agent Behavior at COLM 2026

Review date: 2026-06-27

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting this manuscript for a workshop-oriented review. I evaluated the paper for fit with the Workshop on Agent Behavior at COLM 2026, where the central bar is not only whether an agent performs well, but whether the work advances the scientific study of how agents behave, how that behavior can be measured, and how interventions change behavior.

### Decision: Weak Accept After Minor Revision

The paper is a plausible workshop submission after the latest reframing. Its strongest contribution is a clean behavioral decomposition of forecasting agents: the system's choice to reason, retrieve, defer to a prior, or use a historical analog is treated as an observable behavior rather than an implementation detail. The result is not a large performance breakthrough, but it is a useful behavioral finding: direct source-fitted routing overuses brittle search/graph behavior under transfer, while a constrained reliability router preserves nearly the same behavior as a hand taxonomy and modestly improves Brier score over single deterministic baselines.

The main risk is that reviewers may still see the method as simple routing unless the behavioral claim is made sharper. The paper should explicitly argue that the mechanism-selection policy is the agent behavior under study, and that the intervention is valuable because it exposes and controls over-reasoning. The empirical story is credible, but modest: the gain over the best single baseline is small, the confidence interval against historical analogs overlaps zero, and the AIA-style reproduced baseline is only one comparator rather than the paper's target.

### Consensus Analysis

#### Points of Agreement

- [CONSENSUS-4] The paper fits WAB substantially better when framed around observable mechanism choice and behavioral intervention rather than leaderboard performance.
- [CONSENSUS-4] The strongest empirical contribution is the negative transfer result: a fitted router learns to overuse search/graph behavior and performs worse than simpler alternatives.
- [CONSENSUS-3] ReliabilityRoute is useful, but its novelty is constrained behavioral control from reliability features, not algorithmic complexity.
- [CONSENSUS-3] The AIA-style reproduced baseline should remain in the baseline and related-work role; the paper should not let it dominate the framing.

#### Points of Disagreement

- **Novelty level**: A methodology reviewer would likely view the method as too simple for a main conference, while a workshop reviewer may value the behavioral diagnostic.  
  **Editor's Resolution**: For WAB, acceptability depends on emphasizing the diagnostic/intervention contribution and route-behavior evidence.

- **Performance strength**: One reviewer could argue the Brier gain is too small; another could argue the negative result is the point.  
  **Editor's Resolution**: The paper should sell itself as showing when agentic reasoning behavior fails to transfer, using the reproduced multi-agent baseline as one comparison point.

### Reviewer Summary Matrix

| Dimension | EIC | R1 Methodology | R2 Forecasting/Agents | R3 WAB Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| Recommendation | Weak Accept / Minor Revision | Borderline | Weak Accept | Weak Accept | Borderline Reject if overclaimed |
| Confidence | 4 | 4 | 4 | 3 | 4 |
| Key Strengths | Clear behavioral intervention | Transfer protocol, ablations | Relevant benchmark and baselines | Strong WAB framing potential | Honest negative result |
| Key Weaknesses | Modest empirical effect | Few vintages, threshold fit on one split | Baseline reproduction is approximate | Behavior definition is narrow | Could be dismissed as routing |

## Part 2: Reviewer Reports

### EIC Report

The submission is now aligned with the WAB workshop theme. The most important revision was changing the object of study from "a better forecaster" to "the behavioral policy that chooses which forecasting mechanism gets control." This is a real fit: the paper measures behavior directly with route counts, reports transfer failures in behavior, and frames ReliabilityRoute as an intervention.

The paper should be accepted as a workshop paper if it keeps the claims modest. AIA Forecaster should appear as related work and as motivation for a reproduced comparator, not as the paper's central opponent. The contribution is a reusable behavioral evaluation harness plus a concrete lesson: stronger reasoning modules can degrade performance when the agent invokes them under the wrong reliability conditions.

### R1: Methodology Review

Strengths:

- The no-retune split between the 2024-07-21 human vintage and 2025-10-26 LLM vintage is appropriate for testing transfer.
- The paper reports proper scoring rules, calibration error, bootstrap intervals, oracle headroom, and negative calibration ablations.
- The route-behavior table makes the mechanism-level failure visible.

Weaknesses:

- ReliabilityRoute is threshold-based and tuned on one calibration vintage. This is not enough to establish generality.
- The Brier improvement over HistoricalAnalog is small, and the paired-bootstrap interval slightly overlaps zero.
- Calibration worsens, and the paper has not yet explained whether this is acceptable for downstream forecasting users.

Recommendation: Borderline / Weak Accept if positioned as a workshop diagnostic paper.

### R2: Forecasting and Agent Systems Review

Strengths:

- The paper uses the right benchmark family and includes a reproduced multi-agent forecasting baseline for comparison.
- The source-slice table is informative: ACLED/FRED/Wikipedia favor historical analogs, while DBnomics/YFinance and market-style sources often favor the reproduced multi-agent baseline.
- The negative result on fitted source routing is useful because it demonstrates an agentic over-reasoning failure mode.

Weaknesses:

- The reproduced multi-agent baseline may be too weak to satisfy readers looking for a strong external-system comparison.
- The search loop is deterministic and local; this improves reproducibility but limits claims about live search-enabled LLM agents.
- Metaculus and small market slices remain failure cases.

Recommendation: Weak Accept for WAB; not yet strong enough for a full agent-systems paper without more external baselines.

### R3: Agent Behavior Perspective Review

Strengths:

- The paper now directly studies an agent behavior: choosing whether to reason, retrieve, defer, or use structured analogs.
- ReliabilityRoute is interpretable as a structural behavioral intervention.
- The route diagnostics are exactly the kind of evidence WAB reviewers can use to understand behavior rather than only outcomes.

Weaknesses:

- The behavior definition is narrow. It captures mechanism selection but not richer interaction traces such as query reformulation, tool-use trajectories, deliberation depth, or multi-agent disagreement.
- The paper could more explicitly connect route choices to behavioral failure modes such as over-reasoning and misplaced deference.

Recommendation: Weak Accept after minor revision.

### Devil's Advocate Review

The method may be perceived as "just a taxonomy/router." ReliabilityRoute nearly matches TaxonomyRoute in route counts, so the paper must explain why replacing source names with reliability features matters. The core answer should be transfer and intervention: the method constrains behavior using observable reliability features and avoids a fitted router's label-chasing behavior. Without that argument, the method looks incremental.

The paper should also avoid overselling "agent behavior." It observes one behavior class, mechanism selection, not the full behavior of an interactive LLM agent. This is still a valid WAB contribution, but the limitation must be explicit.

Recommendation: Borderline Reject if overclaimed; Weak Accept if the workshop framing stays precise.

## Part 3: Editorial Synthesis and Revision Roadmap

### Required Revisions

| # | Revision Item | Source | Priority | Estimated Effort |
|---|---|---|---|---|
| R1 | Keep the title, abstract, and introduction centered on behavioral reliability routing, not leaderboard improvement. | EIC/R3 | P1 | 1 hour |
| R2 | Keep the AIA-style reproduced baseline in the baseline/related-work role and avoid making the paper primarily about AIA. | EIC/R2 | P1 | 30 minutes |
| R3 | Preserve the route-behavior diagnostics table and tie it directly to the over-reasoning failure mode. | R1/R3 | P1 | 30 minutes |
| R4 | Make the limitations on mechanism-selection-only behavior explicit. | R3/DA | P1 | 30 minutes |

### Suggested Revisions

| # | Revision Item | Source | Priority | Estimated Effort |
|---|---|---|---|---|
| S1 | Add one sentence in the abstract or conclusion noting that the gain over HistoricalAnalog is modest. | R1 | P2 | 15 minutes |
| S2 | Add a compact route-by-source appendix or supplementary table if page budget allows. | R1/R2 | P2 | 1 hour |
| S3 | Report whether ReliabilityRoute's feature decisions are stable under small threshold perturbations. | R1 | P2 | 2-4 hours |
| S4 | Include a short "Why this is agent behavior" paragraph if reviewers might miss the framing. | R3 | P2 | 30 minutes |

### Final Assessment

The paper has a good WAB story if framed as: "forecasting agents fail not only by producing bad probabilities, but by choosing the wrong behavior under transfer." ReliabilityRoute is deliberately simple, and that simplicity can be a strength for a workshop paper because it makes the behavioral intervention inspectable. The remaining danger is rhetorical: if the paper tries to sound like a state-of-the-art forecasting system, it becomes weak; if it presents itself as a behavioral evaluation and intervention study, it is coherent and useful.
