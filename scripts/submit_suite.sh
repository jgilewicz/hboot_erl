#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: submit_suite.sh <core|controls|ablation-horizon|all> <env>...}"
shift
(($# > 0)) || {
  echo "usage: submit_suite.sh ${suite} <env>..." >&2
  exit 1
}

case "${suite}" in
core) array=0-49 ;;
controls) array=0-19 ;;
ablation-horizon) array=0-29 ;;
all) array=0-99 ;;
*)
  echo "unknown suite '${suite}'" >&2
  exit 1
  ;;
esac

tag="${TAG:-paper-${suite}-v1}"
for env in "$@"; do
  echo "--> ${env}: SUITE=${suite} array=${array} TAG=${tag}"
  SUITE="${suite}" TARGET_ENV="${env}" TAG="${tag}" \
    sbatch --parsable --array="${array}" slurm_run_array.sh
done
