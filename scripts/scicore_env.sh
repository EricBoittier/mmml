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
MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.27.6-GCCcore-13.2.0}"

# The system profile scripts below are not written to survive `set -u`
# (soft_stacks.sh dereferences MODULEPATH before assigning it). Job scripts run
# with `set -u`, where sourcing them aborts the whole job. Relax nounset for the
# duration and restore the caller's setting afterwards.
_mmml_had_nounset=0
case "$-" in
  *u*) _mmml_had_nounset=1 ;;
esac
set +u

# In a Slurm batch script the shell is not a login shell, so neither `module`
# (an lmod shell function) nor MODULEPATH exists. Both are needed: with `module`
# defined but MODULEPATH empty, `module load` finds nothing and fails *silently*,
# and CHARMM then fails to dlopen at runtime.
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

# soft_stacks.sh is what populates MODULEPATH with the easybuild module trees.
if [[ -z "${MODULEPATH:-}" && -r /etc/profile.d/soft_stacks.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/soft_stacks.sh
fi

if command -v module >/dev/null 2>&1; then
  module load "$MMML_SCICORE_TOOLCHAIN" || true
  module load "$MMML_SCICORE_CMAKE" || true
else
  echo "scicore_env: lmod not found; libcharmm will fail to dlopen" >&2
fi

# Warn loudly rather than letting the job run on to an opaque
# "libmpi.so.40 => not found" from deep inside pycharmm.
#
# NB: this file is *sourced*. Never call `exit` here -- it terminates the
# calling job script, not this snippet.
_mmml_found_libmpi=0
_mmml_saved_ifs="$IFS"
IFS=:
for _mmml_dir in ${LD_LIBRARY_PATH:-}; do
  if [[ -e "$_mmml_dir/libmpi.so.40" ]]; then
    _mmml_found_libmpi=1
    break
  fi
done
IFS="$_mmml_saved_ifs"
unset _mmml_dir _mmml_saved_ifs

if [[ "$_mmml_found_libmpi" != "1" ]]; then
  echo "scicore_env: warning: libmpi.so.40 not on LD_LIBRARY_PATH after loading" \
       "'$MMML_SCICORE_TOOLCHAIN' (MODULEPATH=${MODULEPATH:-empty}); CHARMM will fail to load." >&2
fi
unset _mmml_found_libmpi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Restore the caller's nounset setting.
if [[ "$_mmml_had_nounset" == "1" ]]; then
  set -u
fi
unset _mmml_had_nounset

# CHARMM_HOME / CHARMM_LIB_DIR are auto-discovered from setup/charmm; set them
# only to point at an out-of-tree or per-tier CHARMM build.
