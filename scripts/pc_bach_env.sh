#!/usr/bin/env bash
# pc-bach Slurm / shell prolog: OpenMPI 4.1.4 (gcc-12.2.0) for MPI-linked libcharmm.so.
#
# Source before mmml-charmm-mpirun.sh, ensure_charmm_mlpot_limits.sh, or tier rebuilds:
#   source scripts/pc_bach_env.sh
#
# CHARMM_LIB_DIR is normally set per job by ensure_charmm_mlpot_limits.sh (tier cache).
# Override only for manual smoke tests: export MMML_PC_BACH_CHARMM_LIB_DIR=...
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  # Idempotent when modules are already loaded.
  module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4 2>/dev/null || true
  module load charmm/c47a2-gcc-12.2.0-openmpi-4.1.4 2>/dev/null || true
fi

export OPENMPI_ROOT="${OPENMPI_ROOT:-/opt/gcc-12.2.0/openmpi-4.1.4/build}"
export PATH="$OPENMPI_ROOT/bin:${PATH}"
export LD_LIBRARY_PATH="$OPENMPI_ROOT/lib:${LD_LIBRARY_PATH:-}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-cpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [[ -n "${MMML_PC_BACH_CHARMM_LIB_DIR:-}" ]]; then
  export CHARMM_LIB_DIR="$MMML_PC_BACH_CHARMM_LIB_DIR"
fi

export MMML_CLUSTER="${MMML_CLUSTER:-pc-bach}"
