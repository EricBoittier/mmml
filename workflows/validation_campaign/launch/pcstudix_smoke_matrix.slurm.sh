#!/usr/bin/env bash
# Submit with: sbatch workflows/validation_campaign/launch/pcstudix_smoke_matrix.slurm.sh
# Optional runtime filters: KARML_SMOKE_TAG=gpu or KARML_SMOKE_CASE=jaxmd_nve
#SBATCH --job-name=karml-smoke-matrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/validation_campaign/slurm-%x-%j.out
#SBATCH --error=artifacts/validation_campaign/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${KARML_REPO_ROOT:-$HOME/karml}"
cd "$REPO_ROOT"
mkdir -p artifacts/validation_campaign

if [[ -f CHARMMSETUP ]]; then
  # shellcheck disable=SC1091
  source CHARMMSETUP
fi

RUN_ID="${KARML_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"
OUTPUT_ROOT="artifacts/validation_campaign/$RUN_ID/pcstudix/calculator_backend_matrix"
ARGS=()
if [[ -n "${KARML_SMOKE_TAG:-}" ]]; then
  ARGS+=(--tag "$KARML_SMOKE_TAG")
fi
if [[ -n "${KARML_SMOKE_CASE:-}" ]]; then
  ARGS+=(--case "$KARML_SMOKE_CASE")
fi
if [[ "${KARML_SMOKE_STRICT_BLOCKED:-0}" == "1" ]]; then
  ARGS+=(--strict-blocked)
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export KARML_ML_DTYPE="${KARML_ML_DTYPE:-float64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec .venv/bin/python -m karml.validation.smoke_matrix \
  workflows/validation_campaign/pcstudix_smoke_matrix.yaml \
  --output-root "$OUTPUT_ROOT" \
  "${ARGS[@]}"
