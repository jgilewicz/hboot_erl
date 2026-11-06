#!/usr/bin/env bash
# Submit CORE benchmarks on 5 MuJoCo environments (10 seeds, 5 conditions = 250 tasks total)
# Conditions: erl, sc_erl-random, prefix-h200, prefix-td, hboot-td
set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="$(.venv/bin/python -c 'import wandb; print(wandb.api.api_key or "")' 2>/dev/null || true)"
fi

TAG="${TAG:-paper-core-v1}"
ENVS=(
  "HalfCheetah-v5"
  "Hopper-v5"
  "Walker2d-v5"
  "Ant-v5"
  "Swimmer-v5"
)

echo "Submitting CORE benchmark suite (SUITE=core, array=0-49, TAG=${TAG}) across 5 MuJoCo environments..."

for env in "${ENVS[@]}"; do
  echo "--> Submitting ${env} (50 tasks)..."
  SUITE=core TARGET_ENV="${env}" TAG="${TAG}" sbatch --array=0-49 slurm_run_array.sh
done

echo "All 5 MuJoCo environments submitted (250 tasks total)."
