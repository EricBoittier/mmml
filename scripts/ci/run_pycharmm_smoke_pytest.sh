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
# Separate from MD_SYSTEM_SMOKE: builds_ffparams re-reads CGenFF after the
# pbc_nve/nvt cases in that module and segfaults in pycharmm.read.prm.
MD_SYSTEM_FFPARAMS_SMOKE="$ROOT/tests/unit/test_md_system_unified_ffparams.py"
# Deploying scaled LJ parameters rewrites CHARMM's process-global parameter
# state, so it cannot safely share the aggregate remainder process.
SCALED_LJ_CHARMM_SMOKE="$ROOT/tests/unit/test_scaled_lj_charmm_in_the_loop.py"
# This test intentionally launches a nested pytest process after loading
# libcharmm. Running that child beneath mpirun can deadlock in MPI finalization,
# so exercise it once in a genuinely serial parent process.
PYTEST_EXIT_STATUS_SMOKE="$ROOT/tests/unit/test_pytest_exit_status_preserved.py"
# Deliberately re-reads the CGenFF parameters several times to prove the reload
# is non-destructive, so it must own its process for the same reason as the
# scaled-LJ smoke.
PARAM_READ_CONTRACT_SMOKE="$ROOT/tests/functionality/charmm/test_charmm_param_read_contract.py"
# Registers a real MLpot, so it mutates the process-global MLpot/PSF state.
MLPOT_PAIRS_SMOKE="$ROOT/tests/functionality/mlpot/test_mlpot_pair_lists_against_oracle.py"

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
  "$MD_SYSTEM_FFPARAMS_SMOKE"
  "$SCALED_LJ_CHARMM_SMOKE"
  "$PYTEST_EXIT_STATUS_SMOKE"
  "$MLPOT_PAIRS_SMOKE"
)

# tests/conftest.py auto-marks everything under tests/functionality/charmm/ as
# ``charmm_serial`` ("must not run under mpirun -- second PSF/CGENFF read"), and
# MARK_EXPR excludes that marker. So these modules can never run in the loop
# above: listing PARAM_READ_CONTRACT_SMOKE there selected 0 tests and only broke
# the job. They need the opposite treatment -- a serial parent with no mpirun,
# and a mark expression that keeps charmm_serial in.
SERIAL_SMOKE_PATHS=(
  "$PARAM_READ_CONTRACT_SMOKE"
)
SERIAL_MARK_EXPR="${MMML_PYTEST_SERIAL_MARK:-pycharmm and not gpu}"

# JUnit reports per invocation. pytest exits 0 when every selected test skips,
# so the exit status below cannot distinguish "the live suite passed" from "the
# live suite never ran". scripts/ci/check_test_report.py reads these and fails
# when nothing actually passed.
REPORT_DIR="${MMML_PYTEST_REPORT_DIR:-$ROOT/.ci-reports/junit-pycharmm}"
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

# The exit status below is NOT trustworthy on its own. Once libcharmm is loaded
# into the interpreter, CHARMM's Fortran STOP runs at teardown and replaces
# Python's exit status with 0 -- a session with failing tests still exits 0, so
# `|| status=1` never fires. Reproduce with:
#
#   pytest -q <a test that imports pycharmm and then fails>; echo $?   # -> 0
#
# Every invocation therefore also has to leave a JUnit report behind, and
# scripts/ci/check_test_report.py is what actually decides pass/fail.
status=0

# pytest exit 5 is NO_TESTS_COLLECTED -- a configuration result, not a failure.
# It fires when every test in a smoke module is deselected by MARK_EXPR, and it
# turned the whole charmm job red while every test passed: adding
# tests/functionality/charmm/ to the stateful list met tests/conftest.py, which
# auto-marks that entire directory ``charmm_serial`` (_CHARMM_SERIAL_PATH_PREFIXES),
# MARK_EXPR excludes charmm_serial, so pytest selected 0 of 2, exited 5, and
# mpirun surfaced that as non-zero -- "36 passed, 0 failed" and exit 1.
#
# Tolerate it, but say so loudly: a smoke path that selects nothing is dead
# weight and the warning is how it gets noticed. This does not weaken the gate --
# check_test_report.py's --min-passed floor is what proves the suite actually ran.
run_smoke() {  # run_smoke <report-name> <pytest args...>
  local report_name="$1"; shift
  local report="$REPORT_DIR/$report_name.xml"
  local rc=0
  mpirun -np "$MPI_NP" "$MMML_PYTHON" -m pytest --color=yes \
    --junitxml="$report" "$@" || rc=$?
  if [[ "$rc" -eq 5 ]]; then
    echo "::warning::run_pycharmm_smoke_pytest: $report_name selected no tests" \
         "(pytest exit 5) under -m '$MARK_EXPR'; it contributes nothing to this job" >&2
  elif [[ "$rc" -ne 0 ]]; then
    status=1
  fi
  if [[ ! -s "$report" ]]; then
    # A process killed before pytest could write its report leaves no evidence
    # at all, and an absent file would otherwise just shrink the aggregate the
    # gate inspects instead of failing it.
    echo "::error::run_pycharmm_smoke_pytest: no JUnit report from $report_name;" \
         "the run died before pytest could write one" >&2
    status=1
  fi
}

run_serial_smoke() {  # run_serial_smoke <report-name> <pytest args...>
  local report_name="$1"; shift
  local report="$REPORT_DIR/$report_name.xml"
  local rc=0
  "$MMML_PYTHON" -m pytest --color=yes --junitxml="$report" "$@" || rc=$?
  if [[ "$rc" -eq 5 ]]; then
    echo "::warning::run_pycharmm_smoke_pytest: $report_name selected no tests" \
         "(pytest exit 5); it contributes nothing to this job" >&2
  elif [[ "$rc" -ne 0 ]]; then
    status=1
  fi
  if [[ ! -s "$report" ]]; then
    echo "::error::run_pycharmm_smoke_pytest: no JUnit report from $report_name;" \
         "the serial run died before pytest could write one" >&2
    status=1
  fi
}

# Run every module (do not fail-fast) so CI reports the full set of failures
# rather than only the first.
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"; do
  if [[ "$smoke_path" == "$PYTEST_EXIT_STATUS_SMOKE" ]]; then
    continue
  fi
  run_smoke "$(basename "$smoke_path" .py)" -m "$MARK_EXPR" "$smoke_path" "$@"
done

run_serial_smoke "$(basename "$PYTEST_EXIT_STATUS_SMOKE" .py)" \
  -m "$MARK_EXPR" "$PYTEST_EXIT_STATUS_SMOKE" "$@"

# charmm_serial modules: no mpirun, and a mark expression that keeps the marker in.
for smoke_path in "${SERIAL_SMOKE_PATHS[@]}"; do
  run_serial_smoke "$(basename "$smoke_path" .py)" \
    -m "$SERIAL_MARK_EXPR" "$smoke_path" "$@"
done

ignore_args=()
for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}" "${SERIAL_SMOKE_PATHS[@]}"; do
  ignore_args+=("--ignore=$smoke_path")
done

run_smoke remainder -m "$MARK_EXPR" "${ignore_args[@]}" "$@"

exit "$status"
