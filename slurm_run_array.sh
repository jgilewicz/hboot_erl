#!/bin/bash -l
# Truncated fitness evaluation in budget-constrained ERL (MuJoCo v5).
# Envs: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Swimmer-v5, Humanoid-v5
#   (DMC dog-* and MyoSuite myo* are registered too; myo* uses .venv-myosuite)
#
# Suites (SUITE env var, default core); task ID = condition_index * 10 + seed:
#   core             erl, sc_erl-random, prefix-h200, prefix-td, hboot-td  array 0-49
#   controls         rl-only (TD3, no population), sc_erl-always            array 0-19
#   ablation-horizon prefix-h100, prefix-h400, prefix-h800                  array 0-29
#   all              core + controls + ablation-horizon                     array 0-99
#
# Example usage:
#   TARGET_ENV=Humanoid-v5 TAG=paper-core-v1 sbatch --array=0-49 slurm_run_array.sh
#   SUITE=controls TARGET_ENV=Hopper-v5 TAG=paper-controls-v1 sbatch --array=0-19 slurm_run_array.sh
#   ./scripts/submit_suite.sh ablation-horizon HalfCheetah-v5 Hopper-v5

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=erl-bench
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

ENV="${TARGET_ENV:?set TARGET_ENV, e.g. TARGET_ENV=HalfCheetah-v5}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?run sbatch from the checkout or set PROJECT_DIR}}"
TOTAL_STEPS="${TOTAL_STEPS:-1_000_000}"
TAG="${TAG:-paper-benchmark-v1}"
SUITE="${SUITE:-core}"

# Define condition suites: name|overrides
CORE=(
  "erl|algorithm=erl algorithm.eval_mode=full"
  "sc_erl-random|algorithm=erl algorithm.eval_mode=sc_erl_random algorithm.omega=0.79"
  "prefix-h200|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=200"
  "prefix-td|algorithm=erl algorithm.eval_mode=prefix_td"
  "hboot-td|algorithm=erl algorithm.eval_mode=hboot_td"
)
CONTROLS=(
  "rl-only|algorithm=erl algorithm.eval_mode=rl_only"
  "sc_erl-always|algorithm=erl algorithm.eval_mode=sc_erl_always"
)
ABLATION_HORIZON=(
  "prefix-h100|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=100"
  "prefix-h400|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=400"
  "prefix-h800|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=800"
)
case "${SUITE}" in
core) CONDITIONS=("${CORE[@]}") ;;
controls) CONDITIONS=("${CONTROLS[@]}") ;;
ablation-horizon) CONDITIONS=("${ABLATION_HORIZON[@]}") ;;
all) CONDITIONS=("${CORE[@]}" "${CONTROLS[@]}" "${ABLATION_HORIZON[@]}") ;;
*)
  echo "ERROR: Unknown SUITE '${SUITE}'. Choose from: core, controls, ablation-horizon, all" >&2
  exit 1
  ;;
esac

SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
COND_IDX=$((TASK_ID / N_SEEDS))
((COND_IDX < ${#CONDITIONS[@]})) || {
  echo "ERROR: task ${TASK_ID} out of range for SUITE=${SUITE} (${#CONDITIONS[@]} conditions x ${N_SEEDS} seeds)" >&2
  exit 1
}
CONDITION="${CONDITIONS[${COND_IDX}]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
NAME="${CONDITION%%|*}"
read -r -a OVERRIDES <<<"${CONDITION#*|} ${EXTRA:-}"
ENV_SLUG="${ENV//\//_}"
RUN_NAME="${NAME}_${ENV_SLUG}_s${SEED}"

echo "Task ${TASK_ID} | ${NAME} | ${ENV} | seed ${SEED}"

cd "${PROJECT_DIR}"

# Select environment venv (MyoSuite requires dedicated python stack)
if [[ "${ENV}" == myo* ]]; then
  VENV="${PROJECT_DIR}/.venv-myosuite/bin"
  [[ -x "${VENV}/python" ]] || {
    echo "no myosuite venv at ${VENV} - ensure .venv-myosuite is installed" >&2
    exit 1
  }
else
  VENV="${PROJECT_DIR}/.venv/bin"
  [[ -x "${VENV}/python" ]] || {
    echo "no venv at ${VENV} - run 'uv sync' in ${PROJECT_DIR} on the login node" >&2
    exit 1
  }
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$("${VENV}/python" -c 'import wandb; print(wandb.api.api_key or "")' 2>/dev/null || true)"
  export WANDB_API_KEY
fi
: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting or run wandb login}"
export WANDB_MODE="${WANDB_MODE:-online}"
PD_ROOT="/lustre/pd03/${SLURM_JOB_ACCOUNT:-hpc-jakgil6519-1780267238}"
if [[ -d "${PD_ROOT}" && -w "${PD_ROOT}" ]]; then
  export WANDB_DIR="${PD_ROOT}/wandb_logs/${RUN_NAME}"
else
  export WANDB_DIR="${PROJECT_DIR}/wandb_logs/${RUN_NAME}"
fi
mkdir -p logs "${WANDB_DIR}"

if "${VENV}/python" -m train \
  "${OVERRIDES[@]}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  total_steps="${TOTAL_STEPS}" \
  n_envs="${SLURM_CPUS_PER_TASK:-4}" \
  wandb.enabled=true \
  "wandb.name=${RUN_NAME}" \
  "wandb.tags=[${TAG},${NAME},${ENV_SLUG}]" \
  hydra.run.dir="outputs/${RUN_NAME}"; then
  if [[ "${WANDB_MODE}" == offline ]]; then
    for d in "${WANDB_DIR}"/wandb/offline-run-*; do
      [[ -d "$d" ]] && "${VENV}/wandb" sync "$d"
    done
  fi
else
  echo "ERROR: ${RUN_NAME} failed" >&2
  exit 1
fi
