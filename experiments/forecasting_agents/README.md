# Forecasting Agents

Offline prototypes for binary event forecasting systems inspired by AIA Forecaster.

## Systems

- `aia_baseline`: multi-agent search analogue with supervisor-style median aggregation and fixed extremization.
- `evidence_graph_v1`: evidence grouping, question decomposition, specialist forecasts, and logit aggregation.
- `structured_evidence_graph`: evidence graph with stronger weighting for structured time-series evidence.
- `search_llm_loop`: search-enabled reasoning loop that plans subqueries, retrieves cutoff-filtered local/historical evidence, runs specialist reasoning agents, and adjudicates a final probability. Defaults to deterministic offline reasoning; optional OpenAI-compatible live reasoning is available in the dedicated runner.
- `advanced_v1`: blind/market-aware tracks, adversarial review, scenario simulation, error memory, and domain calibration.

## Run

```powershell
python -m experiments.forecasting_agents.run_forecasting_experiment
```

Use a local JSON dataset:

```powershell
python -m experiments.forecasting_agents.run_forecasting_experiment --dataset experiments\forecasting_agents\sample_dataset.json
```

Run tests:

```powershell
python -m pytest experiments\test_forecasting_agents.py
```

## ForecastBench

Run the public matched-target FB-7-21 adapter:

```powershell
python -m experiments.forecasting_agents.run_forecastbench_experiment --question-set 2024-07-21-human --include-superforecasters --run-name forecastbench_fb_7_21_matched_20260625
```

Run the market-source subset:

```powershell
python -m experiments.forecasting_agents.run_forecastbench_experiment --question-set 2024-07-21-human --market-subset --include-superforecasters --run-name forecastbench_fb_market_matched_20260625
```

The adapter downloads public files from `forecastingresearch/forecastbench-datasets` into `experiments/forecastbench_data/raw`.

Run the search-enabled reasoning loop with deterministic offline reasoning:

```powershell
python -m experiments.forecasting_agents.run_search_llm_forecastbench --question-set 2025-10-26-llm --sources fred dbnomics yfinance --out-dir experiments\results\forecastbench_search_llm_2025_10_26
```

Use live OpenAI-compatible reasoning by setting `OPENAI_API_KEY` and adding `--reasoner openai --model <model>`. The runner still uses the same cutoff-filtered retrieval inputs; live model calls only replace the local reasoning judge.

DeepSeek can be used through the same OpenAI-compatible path:

```powershell
$env:DEEPSEEK_API_KEY="..."
python -m experiments.forecasting_agents.run_search_llm_forecastbench --question-set 2025-10-26-llm --limit 40 --reasoner openai --model deepseek-chat --openai-base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --out-dir experiments\results\forecastbench_search_llm_deepseek_smoke
```

## Notes

The bundled fixtures are only smoke-test data. They verify harness behavior, trace output, and metric calculation; they are not evidence that any system beats AIA Forecaster. The next meaningful step is to load a ForecastBench-like export with frozen cutoffs, resolved outcomes, and leakage-checked evidence.
