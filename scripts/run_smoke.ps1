$ErrorActionPreference = "Stop"

python -m pytest experiments/test_forecasting_agents.py
python -m experiments.forecasting_agents.run_forecasting_experiment --out-dir experiments/results/smoke_repro

Write-Host "Smoke checks completed."
