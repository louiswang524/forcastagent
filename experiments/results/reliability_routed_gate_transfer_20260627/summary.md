# Source-Routed Gate Transfer Experiment

Calibration set: `2024-07-21-human`
Evaluation set: `2025-10-26-llm`
Default system: `evidence_graph_v1`

## Learned Source Router

| Source | Selected system |
|---|---|
| acled | `evidence_graph_v1` |
| dbnomics | `evidence_graph_v1` |
| fred | `historical_analog` |
| infer | `search_llm_loop` |
| manifold | `search_llm_loop` |
| metaculus | `aia_baseline` |
| polymarket | `search_llm_loop` |
| wikipedia | `search_llm_loop` |
| yfinance | `evidence_graph_v1` |

## Evaluation Metrics

| System | N | Brier | Log score | Calibration error |
|---|---:|---:|---:|---:|
| taxonomy_routed_gate | 1077 | 0.1846 | 0.5393 | 0.0650 |
| reliability_routed_gate | 1077 | 0.1846 | 0.5394 | 0.0649 |
| historical_analog | 1077 | 0.1876 | 0.5460 | 0.0240 |
| historical_evidence_blend | 1077 | 0.1886 | 0.5505 | 0.0231 |
| search_llm_loop | 1077 | 0.1888 | 0.5524 | 0.0271 |
| hybrid_analog_search | 1077 | 0.1888 | 0.5524 | 0.0271 |
| source_routed_gate | 1077 | 0.1949 | 0.5651 | 0.0797 |
| aia_baseline | 1077 | 0.2009 | 0.5808 | 0.1200 |
| evidence_graph_v1 | 1077 | 0.2048 | 0.5858 | 0.1134 |
| market_only | 1077 | 0.2296 | 0.6406 | 0.1180 |

## Evaluation Source Brier (N)

| Source | reliability_routed_gate | taxonomy_routed_gate | source_routed_gate | historical_analog | search_llm_loop | market_only | aia_baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| acled | 0.1112 (200) | 0.1112 (200) | 0.1148 (200) | 0.1112 (200) | 0.1160 (200) | 0.2500 (200) | 0.1543 (200) |
| dbnomics | 0.2330 (190) | 0.2330 (190) | 0.2644 (190) | 0.2470 (190) | 0.2470 (190) | 0.2500 (190) | 0.2330 (190) |
| fred | 0.2292 (196) | 0.2292 (196) | 0.2292 (196) | 0.2292 (196) | 0.2292 (196) | 0.2500 (196) | 0.2564 (196) |
| infer | 0.0391 (5) | 0.0391 (5) | 0.0529 (5) | 0.0548 (5) | 0.0529 (5) | 0.0548 (5) | 0.0391 (5) |
| manifold | 0.0347 (21) | 0.0347 (21) | 0.0377 (21) | 0.0377 (21) | 0.0377 (21) | 0.0377 (21) | 0.0347 (21) |
| metaculus | 0.2321 (11) | 0.2321 (11) | 0.2321 (11) | 0.2072 (11) | 0.2073 (11) | 0.2072 (11) | 0.2321 (11) |
| polymarket | 0.0170 (70) | 0.0167 (70) | 0.0209 (70) | 0.0209 (70) | 0.0209 (70) | 0.0209 (70) | 0.0167 (70) |
| wikipedia | 0.1821 (192) | 0.1821 (192) | 0.1842 (192) | 0.1821 (192) | 0.1842 (192) | 0.2500 (192) | 0.2006 (192) |
| yfinance | 0.2489 (192) | 0.2489 (192) | 0.2676 (192) | 0.2508 (192) | 0.2509 (192) | 0.2500 (192) | 0.2489 (192) |
