#!/usr/bin/env bash
# scicore Slurm / shell prolog for MPI-linked libcharmm.so.
#
# Source before anything imports pycharmm:
#   source scripts/scicore_env.sh
#
# Why this is needed: rtx4090 compute nodes ship an older userland than the
# toolchain libcharmm.so was built against, so dlopen fails with
#
#   libstdc++.so.6: version `GLIBCXX_3.4.32' not found
#   libmpi.so.40 => not found
#
# foss/2023b supplies both (GCC 13.2 -> GLIBCXX_3.4.32, OpenMPI 4.1.6 ->
# libmpi.so.40) and resolves every libcharmm dependency.
#
# Deliberately no `set -e`: this file is *sourced*, and imposing errexit on the
# calling job script would abort it on the first tolerated non-zero command.

MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2023b}"

# In a Slurm batch script the shell is not a login shell, so `module` -- a shell
# function installed by lmod's init -- does not exist yet. Bootstrap it, or the
# module load below silently does nothing and CHARMM fails to load at runtime.
if ! command -v module >/dev/null 2>&1; then
  for _mmml_lmod_init in \
    "${LMOD_PKG:-/scicore/soft/lmod/lmod}/init/bash" \
    /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash; do
    if [[ -r "$_mmml_lmod_init" ]]; then
      # shellcheck disable=SC1090
      source "$_mmml_lmod_init"
      break
    fi
  done
  unset _mmml_lmod_init
fi

if command -v module >/dev/null 2>&1; then
  module load "$MMML_SCICORE_TOOLCHAIN" 2>/dev/null || true
else
  echo "scicore_env: lmod not found; libcharmm will fail to load" >&2
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# CHARMM_HOME / CHARMM_LIB_DIR are auto-discovered from setup/charmm; set them
# only to point at an out-of-tree or per-tier CHARMM build.
