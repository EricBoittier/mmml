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
MPI_LIVE_ENERGY_SMOKE="$ROOT/tests/charmm_mpi/test_mpi_live_energy.py"
COMP_VELOCITIES_SMOKE="$ROOT/tests/functionality/mlpot/test_comp_velocities_integration.py"
CG_JAXMD_SMOKE="$ROOT/tests/unit/test_cg_jaxmd_unified.py"
DIMER_MODELS_SMOKE="$ROOT/tests/unit/test_dimer_default_models_regression.py"

# CHARMM owns process-global PSF/topology/parameter state.  Any smoke module that
# initializes or rebuilds that state must run in a fresh interpreter: re-reading
# CGenFF after a prior CHARMM build can segfault inside pycharmm.read.prm (the
# cg-jaxmd box builds and the dimer regression each rebuild topology + re-read
# CGenFF).  Keep each stateful module isolated, then run the non-stateful
# remainder.
STATEFUL_SMOKE_PATHS=(
  "$PYCHARMM_RES_SMOKE"
  "$MPI_LIVE_ENERGY_SMOKE"
  "$COMP_VELOCITIES_SMOKE"
  "$CG_JAXMD_SMOKE"
  "$DIMER_MODELS_SMOKE"
)
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"; do
  mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
    -m "$MARK_EXPR" "$smoke_path" "$@"
done

ignore_args=()
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"; do
  ignore_args+=("--ignore=$smoke_path")
done

exec mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
  -m "$MARK_EXPR" \
  "${ignore_args[@]}" \
  "$@"
