#!/bin/bash -l
# Benchmark Suite: Evaluates Fitness Estimation Strategies in Step-Budgeted ERL.
# Supports 15 continuous control environments across 3 suites:
#   - MuJoCo v5 (5 envs): HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Swimmer-v5
#   - DMC Dog (5 envs): dm_control/dog-stand-v0, dog-walk-v0, dog-trot-v0, dog-run-v0, dog-fetch-v0
#   - MyoSuite (5 envs): myoElbowPose1D6MRandom-v0, myoHandReachRandom-v0, myoHandPenTwirlRandom-v0, myoHandObjHoldRandom-v0, myoLegWalk-v0
#
# Modular Suites (controlled via SUITE env var, default: core):
#   - SUITE=core (5 conditions x 10 seeds = 50 tasks, array 0-49):
#       erl (full H=1000), sc_erl-random, prefix-h200, prefix-td, hboot-td
#   - SUITE=ablation-horizon (4 conditions x 10 seeds = 40 tasks, array 0-39):
#       prefix-h100, prefix-h400, prefix-h600, prefix-h800
#   - SUITE=ablation-surrogate (5 conditions x 10 seeds = 50 tasks, array 0-49):
#       prefix-random, hboot-h200, sc_erl-always, ensemble, semarl-td
#   - SUITE=all (14 conditions x 10 seeds = 140 tasks, array 0-139)
#
# Task ID = condition_index * 10 + seed
#
# Example usage:
#   TARGET_ENV=HalfCheetah-v5 sbatch --array=0-49 slurm_run_array.sh            # CORE suite (5 conds x 10 seeds)
#   TARGET_ENV=HalfCheetah-v5 sbatch --array=0-9 slurm_run_array.sh             # ERL baseline only
#   SUITE=all TARGET_ENV=HalfCheetah-v5 sbatch --array=0-139 slurm_run_array.sh # Full matrix

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
case "${SUITE}" in
  core)
    CONDITIONS=(
      "erl|algorithm=erl algorithm.eval_mode=full"
      "sc_erl-random|algorithm=erl algorithm.eval_mode=sc_erl_random algorithm.omega=0.79"
      "prefix-h200|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=200"
      "prefix-td|algorithm=erl algorithm.eval_mode=prefix_td"
      "hboot-td|algorithm=erl algorithm.eval_mode=hboot_td"
    )
    ;;
  ablation-horizon)
    CONDITIONS=(
      "prefix-h100|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=100"
      "prefix-h400|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=400"
      "prefix-h600|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=600"
      "prefix-h800|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=800"
    )
    ;;
  ablation-surrogate)
    CONDITIONS=(
      "prefix-random|algorithm=erl algorithm.eval_mode=prefix_random"
      "hboot-h200|algorithm=erl algorithm.eval_mode=hboot algorithm.prefix_steps=200"
      "sc_erl-always|algorithm=erl algorithm.eval_mode=sc_erl_always"
      "ensemble|algorithm=erl algorithm.eval_mode=ensemble algorithm.k_ensembles=5"
      "semarl-td|algorithm=erl algorithm.eval_mode=semarl_td algorithm.semarl_alpha=1.0 algorithm.semarl_beta=1.0"
    )
    ;;
  all|full)
    CONDITIONS=(
      "erl|algorithm=erl algorithm.eval_mode=full"
      "sc_erl-random|algorithm=erl algorithm.eval_mode=sc_erl_random algorithm.omega=0.79"
      "prefix-h200|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=200"
      "prefix-td|algorithm=erl algorithm.eval_mode=prefix_td"
      "hboot-td|algorithm=erl algorithm.eval_mode=hboot_td"
      "prefix-h100|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=100"
      "prefix-h400|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=400"
      "prefix-h600|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=600"
      "prefix-h800|algorithm=erl algorithm.eval_mode=prefix algorithm.prefix_steps=800"
      "prefix-random|algorithm=erl algorithm.eval_mode=prefix_random"
      "hboot-h200|algorithm=erl algorithm.eval_mode=hboot algorithm.prefix_steps=200"
      "sc_erl-always|algorithm=erl algorithm.eval_mode=sc_erl_always"
      "ensemble|algorithm=erl algorithm.eval_mode=ensemble algorithm.k_ensembles=5"
      "semarl-td|algorithm=erl algorithm.eval_mode=semarl_td algorithm.semarl_alpha=1.0 algorithm.semarl_beta=1.0"
    )
    ;;
  *)
    echo "ERROR: Unknown SUITE '${SUITE}'. Choose from: core, ablation-horizon, ablation-surrogate, all" >&2
    exit 1
    ;;
esac

SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDITION="${CONDITIONS[$((TASK_ID / N_SEEDS))]}"
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

: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting}"
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
