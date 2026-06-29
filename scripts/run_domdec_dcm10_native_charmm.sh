#!/usr/bin/env bash
# Native CHARMM DOMDEC smoke from a prebuilt DCM PSF/CRD.
#
# DOMDec syntax: setup/charmm/doc/domdec.info (Example 1) — domdec ndir on the
# continued ENERGY line; see also energy.info [ domdec-spec ].
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
    "$CHARMM_HOME/bin/charmm" \
    "$CHARMM_HOME/exec/charmm" \
    "$CHARMM_HOME/exec/gnu/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)-exec/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)-exec/bin/charmm" \
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)-exec/exec/charmm" \
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
    "$HOME/.cache/mmml-charmm-build/$(platform_tag)-exec" \
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

If no executable exists, this CHARMM build may be library-only. Build one with:
  bash scripts/rebuild_charmm_native_exec.sh
Then rerun:
  CHARMM_EXE=$MMML_ROOT/setup/charmm/charmm bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
  exit 1
fi

test -s "$PSF" || { echo "Missing PSF: $PSF" >&2; exit 1; }
test -s "$CRD" || { echo "Missing CRD: $CRD" >&2; exit 1; }
test -s "$RTF" || { echo "Missing RTF: $RTF" >&2; exit 1; }
test -s "$PRM" || { echo "Missing PRM: $PRM" >&2; exit 1; }
test -x "$MPIRUN" || { echo "Missing mpirun wrapper: $MPIRUN" >&2; exit 1; }

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

_domdec_strict_c47() {
  # Strict c47 NDIR (1 or >=8 per axis) is opt-in only. Default/auto use NDIR 2 1 1 for np=2.
  case "${DOMDEC_C47_NDIR_RULE:-auto}" in
    1|yes|true|on) echo 1 ;;
    *) echo 0 ;;
  esac
}
DOMDEC_STRICT_C47="$(_domdec_strict_c47)"
SITE_C47=0
if [[ "$CHARMM_EXE" == *c47* ]]; then
  SITE_C47=1
fi

MMML_MPI_NP="${MMML_MPI_NP:-2}"
if ! [[ "$MMML_MPI_NP" =~ ^[1-9][0-9]*$ ]]; then
  echo "MMML_MPI_NP must be a positive integer (got: ${MMML_MPI_NP})" >&2
  exit 1
fi

if [[ "$SITE_C47" == 1 && "$MMML_MPI_NP" -gt 1 && "$MMML_MPI_NP" -lt 8 ]]; then
  cat >&2 <<EOF
Note: site c47 ($CHARMM_EXE) often rejects DOMDEC NDIR 2 1 1 at runtime.
Tier 3 will still attempt np=${MMML_MPI_NP} on the dense l40 prep; for a clean gate use MMML native CHARMM:

  CHARMM_EXE=/path/to/mmml/native/charmm bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
fi

if [[ "$MMML_MPI_NP" -ge 8 && "$DOMDEC_STRICT_C47" != 1 ]]; then
  cat >&2 <<EOF
Note: MMML_MPI_NP=${MMML_MPI_NP} needs a ~152Å prep (c47 np=8 path; dilute, poor PBC images).
Prefer MMML_MPI_NP=2 with MMML native CHARMM on domdec_dcm10_l40 instead.
EOF
fi

if [[ -z "${DOMDEC_NDIR:-}" ]]; then
  if ! DOMDEC_NDIR="$("$PY" -c "
from mmml.utils.domdec_ndir import format_domdec_ndir
print(format_domdec_ndir(${MMML_MPI_NP}, strict_c47_axis_rule=bool(int('${DOMDEC_STRICT_C47}'))))
" 2>&1)"; then
    cat >&2 <<EOF
${DOMDEC_NDIR}

Could not choose DOMDEC NDIR for MMML_MPI_NP=${MMML_MPI_NP}.
For dense l40 tier3 use MMML_MPI_NP=2 (default). c47 np=8 needs DOMDEC_C47_NDIR_RULE=1 and ~152Å prep.
EOF
    exit 1
  fi
fi

_min_box="$("$PY" -c "
from mmml.utils.domdec_ndir import min_domdec_crystal_side_A
print(min_domdec_crystal_side_A(
    ${MMML_MPI_NP}, ${DOMDEC_CUTNB}, ${DOMDEC_GROUP_HALO},
    strict_c47_axis_rule=bool(int('${DOMDEC_STRICT_C47}')),
))
")"
DOMDEC_BOX_SIZE="${DOMDEC_BOX_SIZE:-$CRYSTAL_SIDE}"
if ! "$PY" - <<PY
import sys
side = float("${DOMDEC_BOX_SIZE}")
min_box = float("${_min_box}")
sys.exit(0 if side + 1e-6 >= min_box else 1)
PY
then
  cat >&2 <<EOF
Prep lattice ${DOMDEC_BOX_SIZE}Å is too small for MMML_MPI_NP=${MMML_MPI_NP} DOMDEC domains (need >= ${_min_box}Å per-axis split).

Use a dense liquid-box prep large enough for the domain split (typically BOX_SIZE=40 for np=2).
Do not inflate crystal without re-prepping — that removes PBC images ("IMAGES NEED TO BE PRESENT").

  bash scripts/run_domdec_dcm10_smoke.sh prep
  bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
  exit 1
fi

mkdir -p "$RUN_DIR"
INP="$RUN_DIR/domdec_dcm${N_DCM}.inp"
OUT="$RUN_DIR/domdec_dcm${N_DCM}.out"

if [[ -n "$DOMDEC_CMD" ]]; then
  _domdec_energy_block="${DOMDEC_CMD}
energy"
elif [[ -n "${DOMDEC_ENERGY:-}" ]]; then
  _domdec_energy_block="$DOMDEC_ENERGY"
else
  _domdec_energy_block="$("$PY" -c "
from mmml.utils.domdec_ndir import format_domdec_tier3_energy_block
print(format_domdec_tier3_energy_block(
    ${MMML_MPI_NP},
    cutnb=${DOMDEC_CUTNB},
    strict_c47_axis_rule=bool(int('${DOMDEC_STRICT_C47}')),
))
")"
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

rm -f "$OUT"
"$MPIRUN" "$CHARMM_EXE" -i "$INP" -o "$OUT"
_rc=$?

echo "== native CHARMM output tail =="
tail -n 80 "$OUT" || true

if [[ "$_rc" -ne 0 ]] || grep -qE 'ABNORMAL TERMINATION|BOMLEV \( -2\) IS REACHED' "$OUT" 2>/dev/null; then
  echo "DOMDEC smoke failed (rc=${_rc}): see $OUT" >&2
  if grep -qiE 'must have 1 or >=8|number of nodes|invalid.*ndir|domdec.*error' "$OUT" 2>/dev/null; then
    cat >&2 <<EOF
Likely c47 DOMDEC axis rule: site CHARMM rejects np=2 NDIR 2 1 1.
Build MMML native CHARMM (as_library=OFF) and rerun tier3:

  bash scripts/rebuild_charmm_native_exec.sh
  CHARMM_EXE=$CHARMM_HOME/charmm bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
  elif [[ "$SITE_C47" == 1 && "$MMML_MPI_NP" -gt 1 && "$MMML_MPI_NP" -lt 8 ]]; then
    cat >&2 <<EOF
Site c47 ($CHARMM_EXE) often fails np=${MMML_MPI_NP} DOMDEC. Try MMML native CHARMM:

  bash scripts/rebuild_charmm_native_exec.sh
  CHARMM_EXE=$CHARMM_HOME/charmm bash scripts/run_domdec_dcm10_smoke.sh tier3
EOF
  fi
  exit "${_rc:-1}"
fi

if grep -q 'extraneous characters' "$OUT" 2>/dev/null; then
  echo "DOMDEC/ENERGY command was not parsed cleanly (extraneous-characters warning in $OUT)." >&2
  echo "Usually domdec=OFF in the build (CMake disables DOMDEC when colfft/FFTW is missing)." >&2
  echo "Check input block in ${INP} — then:" >&2
  echo "  bash scripts/verify_charmm_domdec_build.sh $CHARMM_EXE" >&2
  echo "  module load FFTW; export FFTW_ROOT=\${EBROOTFFTW}; bash scripts/rebuild_charmm_native_exec.sh --clean" >&2
  exit 1
fi

if [[ "$MMML_MPI_NP" -gt 1 ]] && ! grep -qiE 'NDIR\s*=' "$OUT" 2>/dev/null; then
  echo "DOMDEC did not activate at np=${MMML_MPI_NP} (no NDIR= line in $OUT)." >&2
  echo "Check ${INP}: energy cutnb ... - / domdec ndir ${DOMDEC_NDIR}" >&2
  exit 1
fi

exit "$_rc"
