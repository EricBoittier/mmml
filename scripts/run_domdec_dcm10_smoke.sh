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
#   BOX_SIZE=40          # prep box side (Å); auto-discovered from newest domdec_dcm*_l* prep
#   DOMDEC_BOX_SIZE=     # override crystal side for tier3 (default: max(BOX_SIZE, np*(cutnb+4)))
#   DOMDEC_NDIR=2 1 1   # override; default avoids c47 Y=2..7 auto-NDIR trap
#   DOMDEC_ENERGY='energy domdec ndir 2 1 1'  # full ENER line override
#   BOX_DIR=$TESTS_ROOT/boxes/domdec_dcm10_l32
#   CHARMM_EXE=/path/to/charmm
#   NATIVE_STATE_CMD='...'  # optional override

set -euo pipefail

PHASE="${1:-all}"
MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
N_DCM="${N_DCM:-10}"
BOX_SIZE="${BOX_SIZE:-40}"
BOX_DIR="${BOX_DIR:-}"
PSF="${PSF:-}"
CRD="${CRD:-}"
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

_infer_box_size_from_dir() {
  local dir="${1%/}"
  local box_json="$dir/box.json"
  if [[ -f "$box_json" ]]; then
    local side
    side="$("$PY" - <<PY
import json
from pathlib import Path
data = json.loads(Path(${box_json@Q}).read_text())
side = data.get("box_side_A")
if side is not None:
    print(float(side))
PY
)" || true
    if [[ -n "$side" ]]; then
      BOX_SIZE="$side"
      return 0
    fi
  fi
  if [[ "$dir" =~ _l([0-9]+)$ ]]; then
    BOX_SIZE="${BASH_REMATCH[1]}"
  fi
}

resolve_domdec_box_artifacts() {
  local boxes_root="$TESTS_ROOT/boxes"
  local pattern="domdec_dcm${N_DCM}_l"

  if [[ -n "$PSF" && -s "$PSF" ]]; then
    BOX_DIR="$(dirname "$PSF")"
    CRD="${CRD:-$BOX_DIR/model.crd}"
    _infer_box_size_from_dir "$BOX_DIR"
    return 0
  fi

  if [[ -n "$BOX_DIR" && -s "$BOX_DIR/model.psf" ]]; then
    PSF="$BOX_DIR/model.psf"
    CRD="${CRD:-$BOX_DIR/model.crd}"
    _infer_box_size_from_dir "$BOX_DIR"
    return 0
  fi

  local default_dir="$boxes_root/${pattern}${BOX_SIZE}"
  if [[ -s "$default_dir/model.psf" ]]; then
    BOX_DIR="$default_dir"
    PSF="$BOX_DIR/model.psf"
    CRD="$BOX_DIR/model.crd"
    _infer_box_size_from_dir "$BOX_DIR"
    return 0
  fi

  local best="" best_mtime=0 d psf mt
  shopt -s nullglob
  for d in "$boxes_root"/${pattern}*/; do
    psf="${d}model.psf"
    if [[ -s "$psf" ]]; then
      if stat --version >/dev/null 2>&1; then
        mt="$(stat -c %Y "$psf")"
      else
        mt="$(stat -f %m "$psf")"
      fi
      if [[ "$mt" -gt "$best_mtime" ]]; then
        best_mtime="$mt"
        best="${d%/}"
      fi
    fi
  done
  shopt -u nullglob

  if [[ -n "$best" ]]; then
    BOX_DIR="$best"
    PSF="$BOX_DIR/model.psf"
    CRD="$BOX_DIR/model.crd"
    _infer_box_size_from_dir "$BOX_DIR"
    echo "Using newest prep box: $BOX_DIR (BOX_SIZE=${BOX_SIZE})" >&2
    return 0
  fi

  BOX_DIR="${BOX_DIR:-$default_dir}"
  PSF="$BOX_DIR/model.psf"
  CRD="$BOX_DIR/model.crd"
  return 1
}

prep_target_dir() {
  BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l${BOX_SIZE}}"
  PSF="$BOX_DIR/model.psf"
  CRD="$BOX_DIR/model.crd"
}

require_domdec_box_artifacts() {
  resolve_domdec_box_artifacts || {
    echo "No DOMDEC prep box found under $TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l*" >&2
    echo "Run: BOX_SIZE=40 $0 prep" >&2
    exit 1
  }
}

prep() {
  prep_target_dir
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
  require_domdec_box_artifacts
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
    NATIVE_STATE_CMD="MMML_MPI_NP=2 PSF='$PSF' CRD='$CRD' BOX_SIZE='$BOX_SIZE' BOX_DIR='$BOX_DIR' N_DCM='$N_DCM' bash '$MMML_ROOT/scripts/run_domdec_dcm10_native_charmm.sh'"
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
