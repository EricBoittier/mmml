#!/usr/bin/env bash
# Run the 15-pair JAX/MBD/xTB dimer scan campaign in a GPU environment on scicore.
#
# Usage:
#   bash scripts/run_dimer_scan_campaign_gpu.sh --multipole-checkpoint <path> --mbd-checkpoint <path> [options...]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Initializing environment from ${REPO_ROOT}..." >&2

# Load modules on scicore if command -v module is available
if command -v module >/dev/null 2>&1; then
  module load GCC/14.2.0 OpenMPI/5.0.7-GCC-14.2.0 CMake/3.31.3-GCCcore-14.2.0 2>/dev/null || true
fi

# Resolve environment python and variables
source "${REPO_ROOT}/scripts/resolve_mmml_env.sh"
mmml_resolve_env "${REPO_ROOT}"

# Setup JAX CUDA variables
if [[ -f "${REPO_ROOT}/scripts/setup_jax_cuda_env.sh" ]]; then
  source "${REPO_ROOT}/scripts/setup_jax_cuda_env.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=========================================================="
echo " Starting JAX/MBD/xTB Dimer Scan Campaign on GPU Node"
echo "=========================================================="
echo " CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES}"
echo " JAX_ENABLE_X64:         ${JAX_ENABLE_X64}"
echo "=========================================================="

exec "${MMML_PYTHON}" "${REPO_ROOT}/scripts/run_dimer_scan_campaign.py" "$@"
