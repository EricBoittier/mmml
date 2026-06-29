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
BOX_SIZE="${BOX_SIZE:-32}"
BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/domdec_dcm${N_DCM}_l${BOX_SIZE}}"
PSF="${PSF:-$BOX_DIR/model.psf}"
CRD="${CRD:-$BOX_DIR/model.crd}"
RUN_DIR="${RUN_DIR:-$TESTS_ROOT/runs/domdec_dcm${N_DCM}_native_l${BOX_SIZE}}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"
CHARMM_HOME="${CHARMM_HOME:-$MMML_ROOT/setup/charmm}"
CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-$CHARMM_HOME}"
RTF="${RTF:-$MMML_ROOT/mmml/data/charmm/top_all36_cgenff.rtf}"
PRM="${PRM:-$MMML_ROOT/mmml/data/charmm/par_all36_cgenff.prm}"
CHARMM_EXE="${CHARMM_EXE:-}"

if [[ -z "$CHARMM_EXE" ]]; then
  for candidate in \
    "$CHARMM_HOME/build/cmake/charmm" \
    "$CHARMM_HOME/charmm" \
    "$CHARMM_HOME/exec/charmm" \
    "$CHARMM_HOME/exec/gnu/charmm" \
    "$MMML_ROOT/setup/charmm/build/cmake/charmm"
  do
    if [[ -x "$candidate" ]]; then
      CHARMM_EXE="$candidate"
      break
    fi
  done
fi

if [[ -z "$CHARMM_EXE" || ! -x "$CHARMM_EXE" ]]; then
  cat >&2 <<EOF
Could not find a CHARMM executable.
Set CHARMM_EXE=/path/to/charmm and retry.
Tried common paths under: $CHARMM_HOME
EOF
  exit 1
fi

test -s "$PSF" || { echo "Missing PSF: $PSF" >&2; exit 1; }
test -s "$CRD" || { echo "Missing CRD: $CRD" >&2; exit 1; }
test -s "$RTF" || { echo "Missing RTF: $RTF" >&2; exit 1; }
test -s "$PRM" || { echo "Missing PRM: $PRM" >&2; exit 1; }
test -x "$MPIRUN" || { echo "Missing mpirun wrapper: $MPIRUN" >&2; exit 1; }

mkdir -p "$RUN_DIR"
INP="$RUN_DIR/domdec_dcm${N_DCM}.inp"
OUT="$RUN_DIR/domdec_dcm${N_DCM}.out"

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

crystal define cubic $BOX_SIZE $BOX_SIZE $BOX_SIZE 90.0 90.0 90.0
crystal build cutoff 15.0 noper 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end

nbonds cutnb 15.0 -
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

domdec on
energy

stop
EOF

echo "== Native CHARMM DOMDEC DCM:${N_DCM} smoke =="
echo "CHARMM_EXE: $CHARMM_EXE"
echo "PSF:        $PSF"
echo "CRD:        $CRD"
echo "INP:        $INP"
echo "OUT:        $OUT"

export CHARMM_HOME CHARMM_LIB_DIR
export LD_LIBRARY_PATH="$CHARMM_LIB_DIR:${LD_LIBRARY_PATH:-}"
export MMML_MPI_NP="${MMML_MPI_NP:-2}"

"$MPIRUN" "$CHARMM_EXE" -i "$INP" -o "$OUT"

echo "== native CHARMM output tail =="
tail -n 80 "$OUT" || true
