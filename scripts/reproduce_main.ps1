$ErrorActionPreference = "Stop"

python -m experiments.forecasting_agents.source_routed_gate_transfer `
  --self-adjust-thresholds `
  --out-dir experiments/results/reliability_routed_gate_transfer_repro

python -m experiments.forecasting_agents.conference_gate_experiment `
  --question-set 2025-10-26-llm `
  --out-dir experiments/results/conference_gate_2025_10_26_repro

python -m experiments.forecasting_agents.paper_quality_analysis `
  --out-dir experiments/results/paper_quality_reliability_repro

python -m experiments.forecasting_agents.multivintage_behavior_analysis `
  --self-adjust-thresholds `
  --walk-forward-adjustment `
  --out-dir experiments/results/multivintage_self_adjust_repro

Write-Host "Main reproducibility runs completed."
