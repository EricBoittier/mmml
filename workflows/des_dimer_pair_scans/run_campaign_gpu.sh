#!/usr/bin/env bash
# Run the des_dimer_pair_scans campaign in the current GPU environment on scicore.
#
# Usage:
#   bash run_campaign_gpu.sh [MAX_JOBS]
#
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"

echo "Initializing environment from ${REPO_ROOT}..." >&2

# Load modules on scicore if command -v module is available
if command -v module >/dev/null 2>&1; then
  module load GCC/14.2.0 OpenMPI/5.0.7-GCC-14.2.0 CMake/3.31.3-GCCcore-14.2.0 2>/dev/null || \
  module load GCC OpenMPI CMake 2>/dev/null || true
fi

# Resolve environment python and variables
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

# Setup JAX CUDA variables
if [[ -f "$REPO_ROOT/scripts/setup_jax_cuda_env.sh" ]]; then
  source "$REPO_ROOT/scripts/setup_jax_cuda_env.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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
echo "=========================================================="

exec "${MMML_PYTHON}" "${WORKFLOW_ROOT}/run_campaign_local.py" \
  --config "${CFG_PATH}" \
  --jobs "$JOBS"
