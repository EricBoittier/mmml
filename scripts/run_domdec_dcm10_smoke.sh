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
#   ./scripts/run_domdec_dcm10_smoke.sh prep-tier3
#   ./scripts/run_domdec_dcm10_smoke.sh tier3
#   ./scripts/run_domdec_dcm10_smoke.sh all
#
# Environment overrides:
#   MMML_ROOT=$HOME/mmml
#   TESTS_ROOT=$HOME/tests
#   N_DCM=10
#   BOX_SIZE=40          # dense prep (validate only); tier3 needs prep-tier3 (~152Å)
#   MMML_MPI_NP=8        # c47 DOMDEC: each NDIR axis must be 1 or >=8
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
  BOX_SIZE="$(_box_crystal_side "${1%/}")"
}

_box_crystal_side() {
  local dir="${1%/}"
  "$PY" - <<PY
import json
import re
from pathlib import Path

d = Path(${dir@Q})
box_json = d / "box.json"
if box_json.is_file():
    side = json.loads(box_json.read_text()).get("box_side_A")
    if side is not None:
        print(float(side))
        raise SystemExit(0)
match = re.search(r"_l([0-9]+)$", d.name)
if match:
    print(float(match.group(1)))
else:
    raise SystemExit(1)
PY
}

resolve_domdec_box_artifacts() {
  local min_side="${1:-0}"
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

  if [[ "$min_side" != "0" && "$min_side" != "0.0" ]]; then
    local picked
    picked="$("$PY" - <<PY
from pathlib import Path
from mmml.utils.domdec_ndir import pick_domdec_prep_dir, _read_prep_box_side_A

picked = pick_domdec_prep_dir(
    Path(${boxes_root@Q}),
    n_dcm=${N_DCM},
    min_side_A=float("${min_side}"),
)
if picked is None:
    raise SystemExit(1)
side = _read_prep_box_side_A(picked)
print(picked)
print(side)
PY
)" || true
    if [[ -n "$picked" ]]; then
      BOX_DIR="$(sed -n '1p' <<< "$picked")"
      BOX_SIZE="$(sed -n '2p' <<< "$picked")"
      PSF="$BOX_DIR/model.psf"
      CRD="$BOX_DIR/model.crd"
      echo "Using tier3 prep box: $BOX_DIR (BOX_SIZE=${BOX_SIZE}Å, min=${min_side}Å)" >&2
      return 0
    fi
    BOX_DIR="${BOX_DIR:-$boxes_root/${pattern}${BOX_SIZE}}"
    PSF="$BOX_DIR/model.psf"
    CRD="$BOX_DIR/model.crd"
    return 1
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

tier3_prep_box_size() {
  local np="${MMML_MPI_NP:-8}"
  "$PY" -c "
from mmml.utils.domdec_ndir import min_domdec_crystal_side_A
import math
side = min_domdec_crystal_side_A(${np}, 15, 4)
print(int(math.ceil(side)))
"
}

prep_tier3() {
  local side
  side="$(tier3_prep_box_size)"
  echo "== DOMDEC DCM:${N_DCM} tier3 prep (box >= ${side}Å for MMML_MPI_NP=${MMML_MPI_NP:-8}) =="
  BOX_SIZE="$side" prep_target_dir
  prep
}

tier3_min_crystal_side() {
  local np="${MMML_MPI_NP:-8}"
  "$PY" -c "
from mmml.utils.domdec_ndir import min_domdec_crystal_side_A
print(min_domdec_crystal_side_A(${np}, 15, 4))
"
}

require_tier3_box_artifacts() {
  local min_side
  min_side="$(tier3_min_crystal_side)"
  resolve_domdec_box_artifacts "$min_side" || {
    echo "No tier3 prep box >= ${min_side}Å under $TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l*" >&2
    echo "Run: bash $0 prep-tier3" >&2
    exit 1
  }
}

require_domdec_box_artifacts() {
  resolve_domdec_box_artifacts 0 || {
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
  if [[ -z "${PSF:-}" || ! -s "$PSF" ]]; then
    require_domdec_box_artifacts
  fi
  echo "== DOMDEC PSF atom-order validation =="
  test -s "$PSF" || {
    echo "Missing PSF: $PSF (run prep first)" >&2
    exit 1
  }
  "$PY" -m mmml.utils.domdec_psf_order "$PSF"
}

tier3() {
  echo "== DOMDEC Tier 3 native-state launch =="
  require_tier3_box_artifacts
  validate
  test -s "$CRD" || {
    echo "Missing CRD: $CRD (run prep first)" >&2
    exit 1
  }
  if [[ -z "${NATIVE_STATE_CMD:-}" ]]; then
    NATIVE_STATE_CMD="MMML_MPI_NP=${MMML_MPI_NP:-8} PSF='$PSF' CRD='$CRD' BOX_SIZE='$BOX_SIZE' BOX_DIR='$BOX_DIR' N_DCM='$N_DCM' bash '$MMML_ROOT/scripts/run_domdec_dcm10_native_charmm.sh'"
  fi
  echo "$NATIVE_STATE_CMD"
  eval "$NATIVE_STATE_CMD"
}

case "$PHASE" in
  prep) prep ;;
  prep-tier3) prep_tier3 ;;
  validate) validate ;;
  tier3) tier3 ;;
  all)
    prep_tier3
    validate
    tier3
    ;;
  *)
    echo "Usage: $0 [prep|prep-tier3|validate|tier3|all]" >&2
    exit 2
    ;;
esac
