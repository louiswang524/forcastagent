# Forecasting Agent Research Plan

## Systems

1. `aia_baseline`: offline analogue of the AIA Forecaster architecture: multiple focused search agents, median supervisor aggregation, disagreement penalty, and fixed extremization.
2. `evidence_graph_v1`: parses evidence into tagged groups, decomposes the question into base-rate/recent/deadline/counterevidence sub-questions, then aggregates specialist forecasts.
3. `advanced_v1`: adds blind and market-aware tracks, adversarial review, scenario simulation, domain calibration, and forecast error memory.

## Hypotheses

- H1: Evidence graph structure improves Brier score over scalar agent ensembling.
- H2: Decomposition helps most on multi-causal questions and can hurt when sub-question assumptions are wrong.
- H3: Counterevidence and resolution-criteria specialists improve calibration by reducing overconfident false positives.
- H4: Market-aware tracks add value only when market liquidity or coverage is weak.
- H5: Domain-specific calibration beats one global extremization coefficient.
- H9: Source/type priors should be learned or calibrated from held-out ForecastBench structure rather than hand-coded.
- H10: A nontrivial part of the gap is aggregation geometry: source-prior, market, and evidence-channel weights need calibration.
- H11: The remaining AIA gap after structural calibration is mainly an information-acquisition gap: dated retrieval and LLM judgment should produce evidence that source priors cannot.
- H16: ForecastBench source priors need time-to-resolution conditioning; repeated targets from one base question are not exchangeable when horizons differ.
- H17: Horizon/source priors must improve a later public question set without retuning before we treat them as a real mechanism rather than benchmark-specific calibration.
- H18: A dated retrieval layer will reduce the remaining AIA gap more than further aggregation tuning.
- H19: Retrieval value will be largest for non-market macro/time-series sources where source/horizon priors are coarse.
- H20: FRED, DBnomics, and YFinance questions need structured freeze-value reasoning: trend, seasonality, volatility, and threshold direction should be parsed into evidence instead of inferred from source priors.
- H21: Calibration learned on human-written ForecastBench questions may not transfer to LLM-written ForecastBench questions; transfer experiments must report train/eval question-set provenance.
- H22: Historical analog retrieval for structured time-series sources can close more of the AIA gap than hand-coded source/horizon priors, provided observations are filtered to dates no later than the forecast due date.
- H23: Equity historical retrieval improves LLM-written YFinance targets but may not transfer to human-written targets; source inclusion should be selected on a calibration set rather than globally enabled.
- H24: A search-enabled LLM loop only improves over historical analog baselines if the reasoning judge adds source-specific interpretation rather than merely re-aggregating retrieved probabilities.
- H25: Global Platt/extremization parameters learned on one ForecastBench vintage can overfit the event mix; calibration should be transfer-tested by source, horizon, and question provenance before being used in the main system.
- H26: The implemented AIA-style baseline is weak mainly because static evidence aggregation underuses structured historical analog evidence.
- H27: Retrieval source policy matters; enabling every historical source can help later LLM-written questions even when a human-written calibration vintage selects FRED-only.
- H28: Analog/evidence blend weight should be selected on a calibration vintage; current evidence favors trusting leakage-checked analogs over graph blending when analogs exist.
- H29: Search-loop graph/evidence blend weight should be selected on a calibration vintage; current evidence favors higher search/retrieval weight than the previous default.
- H30: Source-aware taxonomy routing can improve Brier over the best single deterministic baseline, but direct per-source winner selection overfits across ForecastBench vintages.
- H31: Routed forecasting systems need conditional calibration; taxonomy routing improved Brier but worsened calibration error relative to pure historical analogs.

## Questions

- RQ1: Which specialist agent has the largest marginal contribution?
- RQ2: When should the system defer to market consensus?
- RQ3: Does forecast error memory generalize or overfit to benchmark artifacts?
- RQ4: Is causal/logit aggregation better than median or mean aggregation?
- RQ5: How much of the AIA gap is explainable by source/horizon base rates versus freeze-time evidence retrieval?
- RQ6: Do horizon-conditioned priors transfer across ForecastBench question dates?
- RQ7: Which source families lose the most Brier under no-retune transfer, and does a source-specific parser close that source-local gap?
- RQ8: Is pure historical analog forecasting or a historical/evidence blend more robust across human-written and LLM-written ForecastBench sets?
- RQ9: Which historical retrieval sources survive a calibration-set source-selection gate, and how much performance is lost relative to oracle source selection on the evaluation set?
- RQ10: Does a live LLM judge add calibrated value over the deterministic search loop after controlling for the same retrieved documents and cutoff-date evidence?
- RQ11: Which prompt styles and Platt parameters improve no-retune transfer Brier, and can any prompt/calibration pair beat the historical analog baseline on later ForecastBench sets?
- RQ12: Can source-policy selection be made robust to question-vintage shift, so 2024 calibration does not underselect useful 2025 DBnomics/YFinance analog coverage?
- RQ13: Does the search-enabled loop need a learned gate that defers to pure historical analogs when structured history is available and uses graph/LLM reasoning only for uncovered targets?
- RQ14: Can source-family taxonomy plus nested validation outperform direct per-source routing under no-retune ForecastBench transfer?
- RQ15: Which conditional calibration scheme preserves the taxonomy gate Brier gain while reducing calibration error?

## Near-Term Harness Work

- Add live search adapters behind the existing `EvidenceItem` interface.
- Add LLM-backed specialist agents while preserving deterministic offline tests.
- Add leakage checks for evidence dated after `cutoff_date`.
- Add ablation flags for each component in `advanced_v1`.
- Replace sample fixtures with ForecastBench or a ForecastBench-like local export.
- Run a no-retune transfer check of the horizon-conditioned prior on the next available public ForecastBench question set.
- Add a structured time-series evidence adapter for FRED, DBnomics, and YFinance using only freeze-time-available values.
- Extend the historical analog retriever to YFinance or another non-rate-limited equity data source, then compare pure analog against historical/evidence blending.
- Add a source-selection report that chooses historical retrieval sources on calibration data and evaluates the chosen policy on later ForecastBench sets.
- Run the search-enabled loop with a live LLM reasoner on a small leakage-checked subset and compare against deterministic reasoner, historical analog, and evidence graph baselines.
- Run prompt-style sweeps for the live reasoner (`plain`, `checklist`, `extremized`, `conservative`) on a fixed calibration/evaluation split before committing to a more expensive full benchmark run.
- Add a learned or rule-based analog availability gate: pure analog when structured history exists, graph/search fallback otherwise.
- Split source-policy calibration by source coverage and question provenance before choosing global retrieval sources.
