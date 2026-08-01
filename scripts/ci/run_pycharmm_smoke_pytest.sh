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
MD_SYSTEM_SMOKE="$ROOT/tests/unit/test_md_system_unified.py"

# CHARMM owns process-global PSF/topology/parameter state.  Any smoke module that
# initializes or rebuilds that state must run in a fresh interpreter: re-reading
# CGenFF after a prior CHARMM build can segfault inside pycharmm.read.prm, and a
# leaked pycharmm module handle surfaces as ``'NoneType' has no attribute
# 'set_positions'``.  The end-to-end builders (cg-jaxmd box, dimer regression,
# md-system) each rebuild topology + re-read CGenFF, so isolate every stateful
# module in its own process, then run the non-stateful remainder.
STATEFUL_SMOKE_PATHS=(
  "$PYCHARMM_RES_SMOKE"
  "$MPI_LIVE_ENERGY_SMOKE"
  "$COMP_VELOCITIES_SMOKE"
  "$CG_JAXMD_SMOKE"
  "$DIMER_MODELS_SMOKE"
  "$MD_SYSTEM_SMOKE"
)

# JUnit reports per invocation. pytest exits 0 when every selected test skips,
# so the exit status below cannot distinguish "the live suite passed" from "the
# live suite never ran". scripts/ci/check_test_report.py reads these and fails
# when nothing actually passed.
REPORT_DIR="${MMML_PYTEST_REPORT_DIR:-$ROOT/reports/junit-pycharmm}"
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

# Run every module (do not fail-fast) and remember whether any run failed, so
# CI reports the full set of failures and never green-lights a masked one.
status=0
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"; do
  report_name="$(basename "$smoke_path" .py)"
  mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
    --junitxml="$REPORT_DIR/$report_name.xml" \
    -m "$MARK_EXPR" "$smoke_path" "$@" || status=1
done

ignore_args=()
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"; do
  ignore_args+=("--ignore=$smoke_path")
done

mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
  --junitxml="$REPORT_DIR/remainder.xml" \
  -m "$MARK_EXPR" \
  "${ignore_args[@]}" \
  "$@" || status=1

exit "$status"
