#!/usr/bin/env bash
# Run one dimer-pair 2D scan.
# Usage: job_shell.sh PAIR_TAG
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
PAIR_TAG="${1:?usage: job_shell.sh PAIR_TAG}"

cd "$REPO_ROOT"

# SciCORE toolchain for MPI-linked libcharmm (idempotent if already loaded).
if [[ -f "$REPO_ROOT/scripts/scicore_env.sh" ]]; then
  # shellcheck source=../../../scripts/scicore_env.sh
  source "$REPO_ROOT/scripts/scicore_env.sh"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG="$_cfg_raw"
else
  CFG="$WORKFLOW_ROOT/$_cfg_raw"
fi
export MMML_WORKFLOW_CONFIG="$CFG"

SCAN_PY="$WORKFLOW_ROOT/scripts/run_pair_scan.py"
WRAPPER="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"

# Prefer the CHARMM MPI launcher (np=1). Fall back to bare python for serial
# libcharmm builds (e.g. local macOS --no-mpi).
if [[ "${MMML_DES_SCAN_NO_MPIRUN:-0}" != "1" && -x "$WRAPPER" ]]; then
  exec "$WRAPPER" python "$SCAN_PY" --config "$CFG" --pair "$PAIR_TAG"
fi

exec "${MMML_PYTHON}" "$SCAN_PY" --config "$CFG" --pair "$PAIR_TAG"
