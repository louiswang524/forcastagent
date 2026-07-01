# Forecasting Agent Behavior Artifact

This repository contains the reproducibility artifact for:

**When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing**

The paper studies forecasting-agent behavior on ForecastBench-style binary forecasting tasks. The core intervention, `ReliabilityRoute`, routes each target among deterministic forecasting mechanisms: historical analogs, a reproduced AIA-style baseline, search/evidence-graph reasoning, and market/crowd priors. The artifact includes the code, cached public ForecastBench inputs, and scripts to rerun the main experiments.

## Repository Layout

- `experiments/forecasting_agents/`: forecasting systems, ForecastBench adapter, reliability routers, calibration/ablation scripts, and multi-vintage analysis.
- `experiments/forecastbench_data/raw/`: cached public ForecastBench question and resolution JSON files used in the reported runs.
- `experiments/results/`: created by reproduction scripts; generated results are not committed.
- `scripts/`: convenience scripts for smoke tests and reproducing the reported runs.

## Environment

Python 3.11+ is recommended. The deterministic experiments use the Python standard library. `pytest` is only needed for tests.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

No API key is required for the deterministic results reported in the paper.

## Quick Check

```bash
python -m pytest experiments/test_forecasting_agents.py
python -m experiments.forecasting_agents.run_forecasting_experiment
```

On Windows PowerShell:

```powershell
.\scripts\run_smoke.ps1
```

## Reproduce Main Paper Results

Single-vintage transfer and source diagnostics:

```bash
python -m experiments.forecasting_agents.source_routed_gate_transfer --self-adjust-thresholds --out-dir experiments/results/reliability_routed_gate_transfer_repro
python -m experiments.forecasting_agents.conference_gate_experiment --question-set 2025-10-26-llm --out-dir experiments/results/conference_gate_2025_10_26_repro
python -m experiments.forecasting_agents.paper_quality_analysis --out-dir experiments/results/paper_quality_reliability_repro
```

Multi-vintage walk-forward self-adjusting router:

```bash
python -m experiments.forecasting_agents.multivintage_behavior_analysis --self-adjust-thresholds --walk-forward-adjustment --out-dir experiments/results/multivintage_self_adjust_repro
```

The full multi-vintage run may take several minutes because it evaluates all later LLM vintages. Generated result artifacts are written under `experiments/results/`. They are intentionally omitted from the repository so reviewers can regenerate them from code.

## Data Notes

ForecastBench inputs are public data from `forecastingresearch/forecastbench-datasets`. The adapter will download missing question/resolution files into `experiments/forecastbench_data/raw/`.

Historical time-series caches are intentionally not committed to keep the repository small. If absent, the historical analog code recreates caches under `experiments/forecastbench_data/timeseries_cache/` as needed.

## Anonymous Submission Note

For anonymous review, upload this repository to GitHub, then create an anonymized link using a service such as Anonymous GitHub or OpenReview's preferred anonymous-repository workflow. Avoid adding account-identifying remotes, author names, or commit metadata to the submitted PDF.
