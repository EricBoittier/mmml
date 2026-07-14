#!/usr/bin/env bash
# scicore Slurm / shell prolog for MPI-linked libcharmm.so.
#
# Source before anything imports pycharmm:
#   source scripts/scicore_env.sh
#
# Why this is needed: the compute nodes' default userland is older than the
# toolchain libcharmm.so was built against. Without a module load, dlopen fails:
#
#   libstdc++.so.6: version `GLIBCXX_3.4.32' not found
#   libmpi.so.40 => not found
#
# foss/2023b supplies both (GCC 13.2 -> GLIBCXX_3.4.32, OpenMPI 4.1.6 ->
# libmpi.so.40) and resolves every libcharmm dependency on rtx4090 nodes.
set -euo pipefail

MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2023b}"

if command -v module >/dev/null 2>&1; then
  # Idempotent: reloading an already-loaded module is a no-op.
  module load "$MMML_SCICORE_TOOLCHAIN" 2>/dev/null || true
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# CHARMM_HOME / CHARMM_LIB_DIR are auto-discovered from setup/charmm; set them
# only to point at an out-of-tree or per-tier CHARMM build.
