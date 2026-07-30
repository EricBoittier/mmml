#!/usr/bin/env bash
# Run the des_dimer_pair_scans campaign in the current GPU environment on scicore.
#
# Usage:
#   bash run_campaign_gpu.sh [MAX_JOBS]
#
# CHARMM on scicore is MPI-linked: this script sources scicore_env.sh (foss/2023b)
# and launches each pair under mmml-charmm-mpirun.sh. Do not load a mismatched
# OpenMPI module here — that yields opaque exit-status-1 failures in every pair.
#
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"

echo "Initializing environment from ${REPO_ROOT}..." >&2

# SciCORE: GLIBCXX + libmpi.so.40 for libcharmm (see scripts/scicore_env.sh).
if [[ -f "$REPO_ROOT/scripts/scicore_env.sh" ]]; then
  # shellcheck source=../../scripts/scicore_env.sh
  source "$REPO_ROOT/scripts/scicore_env.sh"
elif command -v module >/dev/null 2>&1; then
  echo "run_campaign_gpu: scicore_env.sh missing; falling back to foss/2023b" >&2
  module load foss/2023b 2>/dev/null || true
fi

# Resolve environment python and variables
# shellcheck source=../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

# Setup JAX CUDA variables
if [[ -f "$REPO_ROOT/scripts/setup_jax_cuda_env.sh" ]]; then
  # shellcheck source=../../scripts/setup_jax_cuda_env.sh
  source "$REPO_ROOT/scripts/setup_jax_cuda_env.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Pair scans need the MPI bootstrap used by md-system / other CHARMM campaigns.
export MMML_USE_CHARMM_MPIRUN="${MMML_USE_CHARMM_MPIRUN:-1}"
export MMML_MPIRUN_WRAPPER="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"

# Default to running 4 parallel pairs
JOBS="${1:-4}"
shift || true

# Config file override support
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
else
  CFG_PATH="$WORKFLOW_ROOT/$_cfg_raw"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"

echo "=========================================================="
echo " Starting Dimer Scan Campaign on GPU Node"
echo "=========================================================="
echo " Config:                 ${CFG_PATH}"
echo " Parallel Pair Jobs:     ${JOBS}"
echo " CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES}"
echo " JAX_ENABLE_X64:         ${JAX_ENABLE_X64}"
echo " CHARMM mpirun wrapper:  ${MMML_MPIRUN_WRAPPER}"
echo "=========================================================="

exec "${MMML_PYTHON}" "${WORKFLOW_ROOT}/run_campaign_local.py" \
  --config "${CFG_PATH}" \
  --jobs "$JOBS"
