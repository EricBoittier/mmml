#!/usr/bin/env bash
# DCM:10 DOMDEC Tier 3 smoke scaffold.
#
# This intentionally avoids Python-side topology construction for np>1. The supported
# direction is:
#   1. prep:     build/certify DCM:10 PSF/CRD with np=1
#   2. validate: offline DOMDEC hydrogen-order check on the PSF
#   3. tier3:    launch a user-provided CHARMM-native/prebuilt-state command
#
# Usage:
#   ./scripts/run_domdec_dcm10_smoke.sh prep
#   ./scripts/run_domdec_dcm10_smoke.sh validate
#   ./scripts/run_domdec_dcm10_smoke.sh tier3
#   ./scripts/run_domdec_dcm10_smoke.sh all
#
# Environment overrides:
#   MMML_ROOT=$HOME/mmml
#   TESTS_ROOT=$HOME/tests
#   N_DCM=10
#   BOX_SIZE=32          # prep box side (Å); tier3 auto-expands crystal if too small for DOMDEC
#   DOMDEC_BOX_SIZE=     # override crystal side for tier3 (default: max(BOX_SIZE, np*(cutnb+4)))
#   DOMDEC_CMD=domdec    # c47: domdec; newer CHARMM may need "domdec on"
#   BOX_DIR=$TESTS_ROOT/boxes/domdec_dcm10_l32
#   CHARMM_EXE=/path/to/charmm
#   NATIVE_STATE_CMD='...'  # optional override

set -euo pipefail

PHASE="${1:-all}"
MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
N_DCM="${N_DCM:-10}"
BOX_SIZE="${BOX_SIZE:-32}"
BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l${BOX_SIZE}}"
PSF="${PSF:-$BOX_DIR/model.psf}"
CRD="${CRD:-$BOX_DIR/model.crd}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"
PY="${MMML_PYTHON:-$MMML_ROOT/.venv/bin/python}"

if [[ ! -d "$MMML_ROOT" ]]; then
  echo "MMML_ROOT not found: $MMML_ROOT" >&2
  exit 1
fi

cd "$MMML_ROOT"

if [[ -f "$MMML_ROOT/scripts/resolve_mmml_env.sh" ]]; then
  # shellcheck source=scripts/resolve_mmml_env.sh
  source "$MMML_ROOT/scripts/resolve_mmml_env.sh"
  mmml_resolve_env "$MMML_ROOT"
  PY="${MMML_PYTHON:-$PY}"
fi

prep() {
  echo "== DOMDEC DCM:${N_DCM} prep =="
  echo "box: $BOX_DIR"
  MMML_MPI_NP=1 "$MPIRUN" liquid-box \
    --composition "DCM:${N_DCM}" \
    --box-size "$BOX_SIZE" \
    --target-density-g-cm3 1.326 \
    --profile dense \
    -o "$BOX_DIR"
  test -s "$PSF"
  test -s "$CRD"
  echo "wrote:"
  echo "  PSF: $PSF"
  echo "  CRD: $CRD"
}

validate() {
  echo "== DOMDEC PSF atom-order validation =="
  test -s "$PSF" || {
    echo "Missing PSF: $PSF (run prep first)" >&2
    exit 1
  }
  "$PY" -m mmml.utils.domdec_psf_order "$PSF"
}

tier3() {
  echo "== DOMDEC Tier 3 native-state launch =="
  validate
  test -s "$CRD" || {
    echo "Missing CRD: $CRD (run prep first)" >&2
    exit 1
  }
  if [[ -z "${NATIVE_STATE_CMD:-}" ]]; then
    NATIVE_STATE_CMD="MMML_MPI_NP=2 PSF='$PSF' CRD='$CRD' BOX_SIZE='$BOX_SIZE' N_DCM='$N_DCM' bash '$MMML_ROOT/scripts/run_domdec_dcm10_native_charmm.sh'"
  fi
  echo "$NATIVE_STATE_CMD"
  eval "$NATIVE_STATE_CMD"
}

case "$PHASE" in
  prep) prep ;;
  validate) validate ;;
  tier3) tier3 ;;
  all)
    prep
    validate
    tier3
    ;;
  *)
    echo "Usage: $0 [prep|validate|tier3|all]" >&2
    exit 2
    ;;
esac
