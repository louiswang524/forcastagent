# Conference Gate Experiment

Question set: `2025-10-26-llm`
Targets: `1077`
Historical coverage: `578`
Sources: `fred dbnomics yfinance`

## Overall Metrics

| System | N | Brier | Log score | Calibration error |
|---|---:|---:|---:|---:|
| historical_analog | 1077 | 0.1876 | 0.5460 | 0.0240 |
| historical_evidence_blend | 1077 | 0.1886 | 0.5505 | 0.0231 |
| search_llm_loop | 1077 | 0.1888 | 0.5524 | 0.0271 |
| hybrid_analog_search | 1077 | 0.1888 | 0.5524 | 0.0271 |
| aia_baseline | 1077 | 0.2009 | 0.5808 | 0.1200 |
| evidence_graph_v1 | 1077 | 0.2048 | 0.5858 | 0.1134 |

## Source Brier (N)

| source | aia_baseline | historical_analog | search_llm_loop | hybrid_analog_search |
|---|---:|---:|---:|---:|
| acled | 0.1543 (200) | 0.1112 (200) | 0.1160 (200) | 0.1160 (200) |
| dbnomics | 0.2330 (190) | 0.2470 (190) | 0.2470 (190) | 0.2470 (190) |
| fred | 0.2564 (196) | 0.2292 (196) | 0.2292 (196) | 0.2292 (196) |
| infer | 0.0391 (5) | 0.0548 (5) | 0.0529 (5) | 0.0529 (5) |
| manifold | 0.0347 (21) | 0.0377 (21) | 0.0377 (21) | 0.0377 (21) |
| metaculus | 0.2321 (11) | 0.2072 (11) | 0.2073 (11) | 0.2073 (11) |
| polymarket | 0.0167 (70) | 0.0209 (70) | 0.0209 (70) | 0.0209 (70) |
| wikipedia | 0.2006 (192) | 0.1821 (192) | 0.1842 (192) | 0.1842 (192) |
| yfinance | 0.2489 (192) | 0.2508 (192) | 0.2509 (192) | 0.2509 (192) |

## Hybrid Gate Decision

| Decision | N | Brier | Log score |
|---|---:|---:|---:|
| historical_analog | 578 | 0.2422 | 0.6749 |
| search_fallback | 499 | 0.1270 | 0.4104 |
