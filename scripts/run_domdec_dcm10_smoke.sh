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
#   NATIVE_STATE_CMD='...' ./scripts/run_domdec_dcm10_smoke.sh tier3
#   ./scripts/run_domdec_dcm10_smoke.sh all
#
# Environment overrides:
#   MMML_ROOT=$HOME/mmml
#   TESTS_ROOT=$HOME/tests
#   N_DCM=10
#   BOX_SIZE=32
#   BOX_DIR=$TESTS_ROOT/boxes/domdec_dcm10_l32
#   NATIVE_STATE_CMD='MMML_MPI_NP=2 ...'

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
    cat >&2 <<EOF
NATIVE_STATE_CMD is required for the next Tier 3 step.

Reason:
  np>1 PyCHARMM topology setup is unsupported on this stack. Start from a
  CHARMM-native/prebuilt state that loads ${PSF} and ${CRD}, establishes PBC and
  DOMDEC, then attaches MLpot.

Suggested placeholders:
  PSF=$PSF
  CRD=$CRD
  BOX_SIZE=$BOX_SIZE
  MMML_MPI_NP=2

Once a native loader exists, run:
  NATIVE_STATE_CMD='<your command>' $0 tier3
EOF
    exit 2
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
