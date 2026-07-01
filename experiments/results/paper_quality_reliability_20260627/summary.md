# Paper Quality Analysis

## Routing and Calibration Ablations

| System | N | Brier | Log score | Calibration error |
|---|---:|---:|---:|---:|
| oracle_question_router | 1077 | 0.1196 | 0.3822 | 0.2369 |
| oracle_source_router | 1077 | 0.1844 | 0.5377 | 0.0634 |
| taxonomy_routed_gate | 1077 | 0.1846 | 0.5393 | 0.0650 |
| reliability_routed_gate | 1077 | 0.1846 | 0.5394 | 0.0649 |
| historical_analog | 1077 | 0.1876 | 0.5460 | 0.0240 |
| taxonomy_global_calibrated | 1077 | 0.1882 | 0.5531 | 0.0670 |
| historical_evidence_blend | 1077 | 0.1886 | 0.5505 | 0.0231 |
| hybrid_analog_search | 1077 | 0.1888 | 0.5524 | 0.0271 |
| search_llm_loop | 1077 | 0.1888 | 0.5524 | 0.0271 |
| source_routed_gate | 1077 | 0.1949 | 0.5651 | 0.0797 |
| taxonomy_source_calibrated | 1077 | 0.2004 | 0.5798 | 0.1025 |
| aia_baseline | 1077 | 0.2009 | 0.5808 | 0.1200 |
| evidence_graph_v1 | 1077 | 0.2048 | 0.5858 | 0.1134 |
| market_only | 1077 | 0.2296 | 0.6406 | 0.1180 |

## Paired Bootstrap Differences

Differences are `reliability_routed_gate - comparator`; negative values favor the reliability router.

| Comparator | Metric | Mean diff | 95% CI | P(diff >= 0) |
|---|---|---:|---:|---:|
| taxonomy_routed_gate | brier | 0.00002 | [0.00000, 0.00006] | 1.000 |
| historical_analog | brier | -0.00296 | [-0.00618, 0.00047] | 0.050 |
| search_llm_loop | brier | -0.00420 | [-0.00776, -0.00059] | 0.011 |
| source_routed_gate | brier | -0.01030 | [-0.01371, -0.00682] | 0.000 |
| aia_baseline | brier | -0.01622 | [-0.02190, -0.01025] | 0.000 |
| taxonomy_global_calibrated | brier | -0.00351 | [-0.00589, -0.00126] | 0.001 |
| taxonomy_source_calibrated | brier | -0.01573 | [-0.02062, -0.01077] | 0.000 |
| taxonomy_routed_gate | log_score | 0.00004 | [0.00000, 0.00012] | 1.000 |
| historical_analog | log_score | -0.00665 | [-0.01398, 0.00072] | 0.041 |
| search_llm_loop | log_score | -0.01298 | [-0.02147, -0.00420] | 0.003 |
| source_routed_gate | log_score | -0.02571 | [-0.03460, -0.01688] | 0.000 |
| aia_baseline | log_score | -0.04138 | [-0.05781, -0.02486] | 0.000 |
| taxonomy_global_calibrated | log_score | -0.01374 | [-0.02291, -0.00532] | 0.001 |
| taxonomy_source_calibrated | log_score | -0.04039 | [-0.05314, -0.02738] | 0.000 |

## Calibration Parameters

| Scope | Slope | Intercept | Train log score |
|---|---:|---:|---:|
| global | 1.30 | -0.10 | 0.4804 |
| source::acled | 1.50 | 0.70 | 0.1892 |
| source::dbnomics | 2.00 | -0.10 | 0.5521 |
| source::fred | 2.00 | -0.50 | 0.4762 |
| source::infer | 1.30 | -0.10 | 0.4804 |
| source::manifold | 1.30 | -0.10 | 0.4804 |
| source::metaculus | 1.30 | -0.10 | 0.4804 |
| source::polymarket | 0.55 | -0.20 | 0.3946 |
| source::wikipedia | 1.75 | 0.60 | 0.5085 |
| source::yfinance | 2.00 | 0.20 | 0.5963 |
