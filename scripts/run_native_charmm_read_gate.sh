#!/usr/bin/env bash
# Native CHARMM cooperative READ control (no PyCHARMM) for MPI read-gate bisect.
#
# Usage:
#   MMML_MPI_NP=2 ./scripts/run_native_charmm_read_gate.sh
#   MMML_MPI_NP=4 ./scripts/run_native_charmm_read_gate.sh --with-restart
#
# Requires a standalone charmm executable (library-only libcharmm.so is not enough):
#   bash scripts/rebuild_charmm_native_exec.sh --clean

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMML_MPI_NP="${MMML_MPI_NP:-1}"
export MMML_MPI_NP

PSF="${PSF:-$ROOT/artifacts/domdec_spatial_smoke/dcm_20mer.psf}"
CRD="${CRD:-$ROOT/artifacts/domdec_spatial_smoke/dcm_20mer.crd}"
RES="${RES:-$ROOT/artifacts/domdec_spatial_smoke/dcm_20mer.res}"
RTF="${RTF:-$ROOT/mmml/data/charmm/top_all36_cgenff.rtf}"
PRM="${PRM:-$ROOT/mmml/data/charmm/par_all36_cgenff.prm}"
WITH_RESTART=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-restart) WITH_RESTART=1; shift ;;
    --psf) PSF="$2"; shift 2 ;;
    --crd) CRD="$2"; shift 2 ;;
    --res) RES="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      echo "run_native_charmm_read_gate: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

CHARMM_HOME="${CHARMM_HOME:-$ROOT/setup/charmm}"
CHARMM_EXE="${CHARMM_EXE:-}"

find_charmm_exe() {
  local candidate
  for candidate in \
    "$CHARMM_EXE" \
    "$CHARMM_HOME/charmm" \
    "$CHARMM_HOME/bin/charmm" \
    "$CHARMM_HOME/exec/charmm" \
    "$CHARMM_HOME/build/cmake/charmm" \
    "$CHARMM_HOME/build/cmake/bin/charmm"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

CHARMM_EXE="$(find_charmm_exe || true)"
if [[ -z "$CHARMM_EXE" || ! -x "$CHARMM_EXE" ]]; then
  cat >&2 <<EOF
run_native_charmm_read_gate: no executable charmm found.

Set CHARMM_EXE=/path/to/charmm or build one:
  bash scripts/rebuild_charmm_native_exec.sh --clean

Tried:
  CHARMM_HOME=$CHARMM_HOME
  CHARMM_EXE=${CHARMM_EXE:-<unset>}
EOF
  exit 1
fi

for f in "$PSF" "$RTF" "$PRM"; do
  if [[ ! -f "$f" ]]; then
    echo "run_native_charmm_read_gate: missing file: $f" >&2
    exit 1
  fi
done

OUT_DIR="$ROOT/artifacts/domdec_spatial_smoke/native_read_gate"
mkdir -p "$OUT_DIR"
INP="$OUT_DIR/read_gate.inp"
OUT="$OUT_DIR/read_gate_np${MMML_MPI_NP}.out"

if [[ "$WITH_RESTART" == 1 ]]; then
  if [[ ! -f "$RES" ]]; then
    echo "run_native_charmm_read_gate: restart not found: $RES" >&2
    exit 1
  fi
  cat >"$INP" <<EOF
* native read gate (restart)
*
bomlev -2
read rtf card name $RTF
read param card name $PRM
read psf card name $PSF
open read unit 20 name $RES
read restart unit 20
close unit 20
energy
stop
EOF
else
  if [[ ! -f "$CRD" ]]; then
    echo "run_native_charmm_read_gate: CRD not found: $CRD" >&2
    exit 1
  fi
  cat >"$INP" <<EOF
* native read gate (psf-crd)
*
bomlev -2
read rtf card name $RTF
read param card name $PRM
read psf card name $PSF
read coor card name $CRD
energy
stop
EOF
fi

echo "run_native_charmm_read_gate: CHARMM_EXE=$CHARMM_EXE np=$MMML_MPI_NP" >&2
echo "run_native_charmm_read_gate: inp=$INP out=$OUT" >&2

"$ROOT/scripts/mmml-charmm-mpirun.sh" "$CHARMM_EXE" -i "$INP" -o "$OUT"

echo "run_native_charmm_read_gate: grep NATOM / energy from $OUT" >&2
grep -E 'NATOM|ENER|ABNORMAL|PSF has|Total energy' "$OUT" | tail -30 || true
