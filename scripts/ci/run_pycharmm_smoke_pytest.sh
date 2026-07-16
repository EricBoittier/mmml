#!/usr/bin/env bash
# CHARMM smoke pytest for CI (CPU, no JAX GPU wheels).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

if [[ -f "$ROOT/CHARMMSETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/CHARMMSETUP"
fi

if [[ ! -f "${CHARMM_LIB_DIR:-}/libcharmm.so" ]]; then
  echo "ci/run_pycharmm_smoke_pytest: libcharmm.so missing; run scripts/ci/setup_charmm_lib.sh" >&2
  exit 1
fi

MARK_EXPR="${MMML_PYTEST_MARK:-pycharmm and not gpu and not charmm_serial}"
MPI_NP="${MMML_MPI_NP:-1}"
PYCHARMM_RES_SMOKE="$ROOT/tests/functionality/pycharmmETC/test_res.py"

# CHARMM owns process-global PSF/topology/parameter state.  setupRes reads CGenFF
# parameters and can segfault if it follows other live tests in the same process,
# even though it is reliable in a fresh rank.  Keep that smoke coverage, but give
# it its own MPI process before running the remaining selection.
mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
  -m "$MARK_EXPR" "$PYCHARMM_RES_SMOKE" "$@"

exec mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
  -m "$MARK_EXPR" --ignore="$PYCHARMM_RES_SMOKE" "$@"
