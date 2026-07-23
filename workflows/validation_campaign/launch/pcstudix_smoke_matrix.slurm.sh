#!/usr/bin/env bash
# Submit with: sbatch workflows/validation_campaign/launch/pcstudix_smoke_matrix.slurm.sh
# Optional runtime filters: MMML_SMOKE_TAG=gpu or MMML_SMOKE_CASE=jaxmd_nve
#SBATCH --job-name=mmml-smoke-matrix
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

REPO_ROOT="${MMML_REPO_ROOT:-$HOME/mmml}"
cd "$REPO_ROOT"
mkdir -p artifacts/validation_campaign

if [[ -f CHARMMSETUP ]]; then
  # shellcheck disable=SC1091
  source CHARMMSETUP
fi

RUN_ID="${MMML_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"
OUTPUT_ROOT="artifacts/validation_campaign/$RUN_ID/pcstudix/calculator_backend_matrix"
ARGS=()
if [[ -n "${MMML_SMOKE_TAG:-}" ]]; then
  ARGS+=(--tag "$MMML_SMOKE_TAG")
fi
if [[ -n "${MMML_SMOKE_CASE:-}" ]]; then
  ARGS+=(--case "$MMML_SMOKE_CASE")
fi
if [[ "${MMML_SMOKE_STRICT_BLOCKED:-0}" == "1" ]]; then
  ARGS+=(--strict-blocked)
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_ML_DTYPE="${MMML_ML_DTYPE:-float64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec .venv/bin/python -m mmml.validation.smoke_matrix \
  workflows/validation_campaign/pcstudix_smoke_matrix.yaml \
  --output-root "$OUTPUT_ROOT" \
  "${ARGS[@]}"
