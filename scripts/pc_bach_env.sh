#!/usr/bin/env bash
# pc-bach Slurm / shell prolog: OpenMPI 4.1.4 (gcc-12.2.0) for MPI-linked libcharmm.so.
#
# Source before karml-charmm-mpirun.sh, ensure_charmm_mlpot_limits.sh, or tier rebuilds:
#   source scripts/pc_bach_env.sh
#
# CHARMM_LIB_DIR is normally set per job by ensure_charmm_mlpot_limits.sh (tier cache).
# Override only for manual smoke tests: export KARML_PC_BACH_CHARMM_LIB_DIR=...
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  # Idempotent when modules are already loaded.
  module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4 2>/dev/null || true
  module load charmm/c47a2-gcc-12.2.0-openmpi-4.1.4 2>/dev/null || true
fi

export OPENMPI_ROOT="${OPENMPI_ROOT:-/opt/gcc-12.2.0/openmpi-4.1.4/build}"
export PATH="/opt/gcc-12.2.0/cmake-3.25.1/bin:$OPENMPI_ROOT/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/gcc-12.2.0/build/lib64:$OPENMPI_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Cluster /opt FFTW is static-only; linking libcharmm.so needs -fPIC (user-local build).
_KARML_FFTW_PIC="${KARML_FFTW_ROOT:-${HOME}/.local/fftw-3.3.10-pic}"
if [[ -f "${_KARML_FFTW_PIC}/lib/libfftw3.so" ]]; then
  export FFTW_ROOT="$_KARML_FFTW_PIC"
  export CMAKE_PREFIX_PATH="${FFTW_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="${FFTW_ROOT}/lib:${LD_LIBRARY_PATH}"
fi
unset _KARML_FFTW_PIC

# gcc/openmpi modules on pc-bach may export MPI_CXX=.../mpixx (typo); use real wrappers.
export MPI_CC="${OPENMPI_ROOT}/bin/mpicc"
export MPI_CXX="${OPENMPI_ROOT}/bin/mpicxx"
export MPI_FC="${OPENMPI_ROOT}/bin/mpifort"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export KARML_MLPOT_DEVICE="${KARML_MLPOT_DEVICE:-cpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [[ -n "${KARML_PC_BACH_CHARMM_LIB_DIR:-}" ]]; then
  export CHARMM_LIB_DIR="$KARML_PC_BACH_CHARMM_LIB_DIR"
fi

export KARML_CLUSTER="${KARML_CLUSTER:-pc-bach}"
