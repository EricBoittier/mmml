#!/usr/bin/env bash
# Native CHARMM DOMDEC smoke from a prebuilt DCM PSF/CRD.
#
# This is intentionally CHARMM-native: it does not use PyCHARMM topology setup.
# It reads RTF/PRM/PSF/CRD in CHARMM, establishes a cubic PBC box, turns DOMDEC on,
# and runs ENER. Use this as the first np>1 gate before any MLpot attachment.

set -euo pipefail

MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
N_DCM="${N_DCM:-10}"
BOX_SIZE="${BOX_SIZE:-40}"
# Effective nonbond cutoff for DOMDEC domain sizing (cutnb + 2*max_group_radius).
DOMDEC_CUTNB="${DOMDEC_CUTNB:-15}"
DOMDEC_GROUP_HALO="${DOMDEC_GROUP_HALO:-4}"
DOMDEC_CMD="${DOMDEC_CMD:-}"  # optional legacy; prefer DOMDEC_NDIR / DOMDEC_ENERGY
BOX_DIR="${BOX_DIR:-}"
PSF="${PSF:-}"
CRD="${CRD:-}"
if [[ -z "$BOX_DIR" && -n "$PSF" ]]; then
  BOX_DIR="$(dirname "$PSF")"
fi
if [[ -z "$PSF" && -n "$BOX_DIR" ]]; then
  PSF="$BOX_DIR/model.psf"
  CRD="${CRD:-$BOX_DIR/model.crd}"
fi
if [[ -z "$PSF" ]]; then
  BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l${BOX_SIZE}}"
  PSF="$BOX_DIR/model.psf"
  CRD="${CRD:-$BOX_DIR/model.crd}"
fi
_box_tag="$(basename "${BOX_DIR:-domdec_dcm${N_DCM}_l${BOX_SIZE}}")"
RUN_DIR="${RUN_DIR:-$TESTS_ROOT/runs/${_box_tag}_native}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"
CHARMM_HOME="${CHARMM_HOME:-$MMML_ROOT/setup/charmm}"
CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-$CHARMM_HOME}"
CHARMM_BUILD_DIR="${CHARMM_BUILD_DIR:-}"
RTF="${RTF:-$MMML_ROOT/mmml/data/charmm/top_all36_cgenff.rtf}"
PRM="${PRM:-$MMML_ROOT/mmml/data/charmm/par_all36_cgenff.prm}"
CHARMM_EXE="${CHARMM_EXE:-}"

platform_tag() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    darwin) echo "darwin-${arch}" ;;
    linux) echo "linux-${arch}" ;;
    *) echo "${os}-${arch}" ;;
  esac
}

find_charmm_exe() {
  local candidate build_root
  for candidate in \
    "$CHARMM_EXE" \
    "$CHARMM_BUILD_DIR/charmm" \
    "$CHARMM_BUILD_DIR/bin/charmm" \
    "$CHARMM_BUILD_DIR/exec/charmm" \
    "$CHARMM_HOME/build/cmake/charmm" \
    "$CHARMM_HOME/build/cmake/bin/charmm" \
    "$CHARMM_HOME/charmm" \
    "$CHARMM_HOME/bin/charmm" \
    "$CHARMM_HOME/exec/charmm" \
    "$CHARMM_HOME/exec/gnu/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)/bin/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)/exec/charmm"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  for build_root in \
    "$CHARMM_BUILD_DIR" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)" \
    "$CHARMM_HOME/build/cmake"
  do
    if [[ -d "$build_root" ]]; then
      candidate="$(command find "$build_root" -type f -name charmm -perm -111 -print -quit 2>/dev/null || true)"
      if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

CHARMM_EXE="$(find_charmm_exe || true)"

if [[ -z "$CHARMM_EXE" || ! -x "$CHARMM_EXE" ]]; then
  cat >&2 <<EOF
Could not find a CHARMM executable.
Set CHARMM_EXE=/path/to/charmm and retry.

Tried common paths under:
  CHARMM_HOME=$CHARMM_HOME
  CHARMM_BUILD_DIR=${CHARMM_BUILD_DIR:-<unset>}
  $HOME/.cache/mmml-charmm-build/$(platform_tag)

Quick locate:
  command find "$HOME/.cache/mmml-charmm-build" "$CHARMM_HOME" -type f -name charmm -perm -111 2>/dev/null

If no executable exists, this CHARMM build may be library-only. Reconfigure/build a
native CHARMM executable with the same MPI/DOMDEC stack, then rerun:
  CHARMM_EXE=/path/to/charmm bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
  exit 1
fi

test -s "$PSF" || { echo "Missing PSF: $PSF" >&2; exit 1; }
test -s "$CRD" || { echo "Missing CRD: $CRD" >&2; exit 1; }
test -s "$RTF" || { echo "Missing RTF: $RTF" >&2; exit 1; }
test -s "$PRM" || { echo "Missing PRM: $PRM" >&2; exit 1; }
test -x "$MPIRUN" || { echo "Missing mpirun wrapper: $MPIRUN" >&2; exit 1; }

domdec_min_box_size() {
  local np="$1" cutnb="$2" halo="$3"
  echo $(( np * (cutnb + halo) ))
}

_resolve_python() {
  if [[ -n "${MMML_PYTHON:-}" && -x "${MMML_PYTHON}" ]]; then
    echo "$MMML_PYTHON"
  elif [[ -x "${MMML_ROOT}/.venv/bin/python" ]]; then
    echo "${MMML_ROOT}/.venv/bin/python"
  else
    command -v python3 || command -v python
  fi
}

PY="$(_resolve_python)"

_read_crystal_side() {
  local box_json="${1:-}"
  if [[ -z "$box_json" || ! -f "$box_json" ]]; then
    return 1
  fi
  "$PY" - <<PY
import json
from pathlib import Path
data = json.loads(Path(${box_json@Q}).read_text())
side = data.get("box_side_A")
if side is not None:
    print(float(side))
PY
}

CRYSTAL_SIDE="${CRYSTAL_SIDE:-}"
if [[ -z "$CRYSTAL_SIDE" ]]; then
  _box_json="${BOX_DIR:-$(dirname "$PSF")}/box.json"
  CRYSTAL_SIDE="$(_read_crystal_side "$_box_json" 2>/dev/null || true)"
fi
CRYSTAL_SIDE="${CRYSTAL_SIDE:-$BOX_SIZE}"

MMML_MPI_NP="${MMML_MPI_NP:-8}"
if ! [[ "$MMML_MPI_NP" =~ ^[1-9][0-9]*$ ]]; then
  echo "MMML_MPI_NP must be a positive integer (got: ${MMML_MPI_NP})" >&2
  exit 1
fi

if [[ -z "${DOMDEC_NDIR:-}" ]]; then
  if ! DOMDEC_NDIR="$("$PY" -c "
from mmml.utils.domdec_ndir import format_domdec_ndir
print(format_domdec_ndir(${MMML_MPI_NP}))
" 2>&1)"; then
    cat >&2 <<EOF
${DOMDEC_NDIR}

c47 DOMDEC rejects 2–7 MPI nodes per axis (each NDIR axis must be 1 or >= 8).
For np>1 Tier 3 smoke on site c47, use at least MMML_MPI_NP=8:

  MMML_MPI_NP=8 bash scripts/run_domdec_dcm10_smoke.sh tier3

With cutnb=${DOMDEC_CUTNB}, the crystal side must be >= MMML_MPI_NP * (cutnb + ${DOMDEC_GROUP_HALO}) Å
(≈ $(( MMML_MPI_NP * (DOMDEC_CUTNB + DOMDEC_GROUP_HALO) ))Å for np=${MMML_MPI_NP}).
EOF
    exit 1
  fi
fi

_min_box="$(domdec_min_box_size "$MMML_MPI_NP" "$DOMDEC_CUTNB" "$DOMDEC_GROUP_HALO")"
DOMDEC_BOX_SIZE="${DOMDEC_BOX_SIZE:-$CRYSTAL_SIDE}"
DOMDEC_BOX_SIZE="$("$PY" - <<PY
import math
side = float("${DOMDEC_BOX_SIZE}")
min_box = float("${_min_box}")
print(max(side, min_box))
PY
)"
if "$PY" - <<PY
import sys
side = float("${CRYSTAL_SIDE}")
min_box = float("${_min_box}")
sys.exit(0 if side >= min_box else 1)
PY
then
  :
else
  echo "DOMDEC crystal side ${CRYSTAL_SIDE}Å too small for MMML_MPI_NP=${MMML_MPI_NP} and cutnb=${DOMDEC_CUTNB} (need >= ${_min_box}Å); using ${DOMDEC_BOX_SIZE}Å." >&2
fi
if "$PY" - <<PY
import sys
side = float("${DOMDEC_BOX_SIZE}")
prep = float("${CRYSTAL_SIDE}")
sys.exit(0 if abs(side - prep) < 0.01 else 1)
PY
then
  :
else
  echo "Note: crystal uses ${DOMDEC_BOX_SIZE}Å but prep lattice was ${CRYSTAL_SIDE}Å (expanded for DOMDEC)." >&2
fi

mkdir -p "$RUN_DIR"
INP="$RUN_DIR/domdec_dcm${N_DCM}.inp"
OUT="$RUN_DIR/domdec_dcm${N_DCM}.out"

if [[ -n "$DOMDEC_CMD" ]]; then
  _domdec_energy_block="${DOMDEC_CMD}
energy"
else
  DOMDEC_ENERGY="${DOMDEC_ENERGY:-energy domdec ndir ${DOMDEC_NDIR}}"
  _domdec_energy_block="$DOMDEC_ENERGY"
fi

cat > "$INP" <<EOF
* DCM:${N_DCM} native CHARMM DOMDEC smoke
*
bomlev -2
wrnlev 5
prnlev 5

read rtf card name $RTF
read param card name $PRM
read psf card name $PSF
read coor card name $CRD

crystal define cubic $DOMDEC_BOX_SIZE $DOMDEC_BOX_SIZE $DOMDEC_BOX_SIZE 90.0 90.0 90.0
crystal build cutoff 15.0 noper 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end

nbonds cutnb ${DOMDEC_CUTNB}.0 -
  ctonnb 10.83 -
  ctofnb 14.17 -
  eps 1.0 -
  cdie -
  atom -
  vatom -
  fswitch -
  vfswitch -
  nbxmod 5 -
  cutim 15.0 -
  ctexnb 15.0 -
  inbfrq 50 -
  imgfrq 50

$_domdec_energy_block

stop
EOF

echo "== Native CHARMM DOMDEC DCM:${N_DCM} smoke =="
echo "CHARMM_EXE: $CHARMM_EXE"
echo "PSF:        $PSF"
echo "CRD:        $CRD"
echo "DOMDEC:     MMML_MPI_NP=${MMML_MPI_NP} ndir=${DOMDEC_NDIR} crystal=${DOMDEC_BOX_SIZE}Å"
echo "INP:        $INP"
echo "OUT:        $OUT"

export CHARMM_HOME CHARMM_LIB_DIR
export LD_LIBRARY_PATH="$CHARMM_LIB_DIR:${LD_LIBRARY_PATH:-}"

"$MPIRUN" "$CHARMM_EXE" -i "$INP" -o "$OUT"
_rc=$?

echo "== native CHARMM output tail =="
tail -n 80 "$OUT" || true

if [[ "$_rc" -ne 0 ]] || grep -qE 'ABNORMAL TERMINATION|BOMLEV \( -2\) IS REACHED' "$OUT" 2>/dev/null; then
  echo "DOMDEC smoke failed (rc=${_rc}): see $OUT" >&2
  exit "${_rc:-1}"
fi
exit "$_rc"
