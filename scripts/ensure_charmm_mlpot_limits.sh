#!/usr/bin/env bash
# Ensure libcharmm.so max_Npr tier fits the ML atom count for this job.
# Usage: ensure_charmm_mlpot_limits.sh --n-ml 2660 [--pbc]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_ML=""
PBC=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --n-ml N_ML_ATOMS [--pbc] [--dry-run]

Selects the smallest local CHARMM build tier (default/large/xlarge/xxlarge), patches
setup/api/api_func.F90 max_Npr when needed, and rebuilds into:
  \${CHARMM_BUILD_DIR:-\$HOME/.cache/mmml-charmm-build}/tier_\${MAX_NPR}_nodomdec

Tier libs are built with rebuild_charmm_mlpot.sh --no-domdec (MPI MLpot, np=1).

Use --pbc for periodic systems (``mlpot_update`` image ML pairs).

Exports CHARMM_LIB_DIR to the tier lib directory for child processes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-ml) N_ML="${2:?--n-ml requires value}"; shift 2 ;;
    --pbc) PBC=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$N_ML" ]]; then
  echo "ERROR: --n-ml is required" >&2
  usage >&2
  exit 2
fi

read -r TIER TARGET PATCH_F90 <<<"$(
  python3 - "$N_ML" "$ROOT" "$PBC" <<'PY'
import sys
from pathlib import Path

n_ml = int(sys.argv[1])
root = Path(sys.argv[2])
pbc = bool(int(sys.argv[3]))
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
    select_npr_tier,
    tier_max_npr,
    charmm_mlpot_limits_from_source,
)

tier = select_npr_tier(n_ml, pbc=pbc)
target = tier_max_npr(tier)
parsed = charmm_mlpot_limits_from_source()
f90 = parsed[2] if parsed else root / "setup" / "api" / "api_func.F90"
print(tier, target, f90)
PY
)"

echo "ML atoms=${N_ML} pbc=${PBC} -> tier=${TIER} max_Npr=${TARGET}"

PATCH_DST="${PATCH_F90}"
if [[ ! -f "$PATCH_DST" ]]; then
  PATCH_DST="$ROOT/setup/api/api_func.F90"
fi

CURRENT_NPR="$(grep -E 'max_Npr\s*=' "$PATCH_DST" | head -1 | grep -oE '[0-9]+' || true)"
if [[ -z "$CURRENT_NPR" ]]; then
  echo "ERROR: could not read max_Npr from $PATCH_DST" >&2
  exit 1
fi

if [[ "$CURRENT_NPR" -lt "$TARGET" ]]; then
  echo "Patching max_Npr ${CURRENT_NPR} -> ${TARGET} in ${PATCH_DST}"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "(dry-run) would patch ${PATCH_DST}"
  else
    python3 - "$PATCH_DST" "$TARGET" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
text = path.read_text(encoding="utf-8")
new_text, n = re.subn(
    r"(max_Npr\s*=\s*)\d+",
    rf"\g<1>{target}",
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"could not patch max_Npr in {path}")
path.write_text(new_text, encoding="utf-8")
PY
  fi
fi

BUILD_ROOT="${CHARMM_BUILD_DIR:-$HOME/.cache/mmml-charmm-build}"
TIER_DIR="${BUILD_ROOT}/tier_${TARGET}_nodomdec"
LIB_DIR="${TIER_DIR}/lib"
LOCK_FILE="${BUILD_ROOT}/.tier_${TARGET}_nodomdec.build.lock"
mkdir -p "$LIB_DIR" "$BUILD_ROOT"

NEED_BUILD=0
if [[ ! -f "${LIB_DIR}/libcharmm.so" ]]; then
  NEED_BUILD=1
elif [[ "$PATCH_DST" -nt "${LIB_DIR}/libcharmm.so" ]]; then
  NEED_BUILD=1
fi

_run_tier_build() {
  CMAKE_BUILD_DIR="${CHARMM_CMAKE_BUILD_DIR:-${SLURM_TMPDIR:-/tmp}/mmml-charmm-cmake-${TARGET}-$$}"
  echo "Building CHARMM MLpot tier ${TIER} (max_Npr=${TARGET}); cmake in ${CMAKE_BUILD_DIR}"
  rm -rf "$CMAKE_BUILD_DIR" 2>/dev/null || true
  mkdir -p "$CMAKE_BUILD_DIR"
  CHARMM_BUILD_DIR="$CMAKE_BUILD_DIR" "$ROOT/scripts/rebuild_charmm_mlpot.sh" --clean --no-domdec
  mkdir -p "$LIB_DIR"
  cp -f "${CHARMM_HOME:-$ROOT/setup/charmm}/libcharmm.so" "${LIB_DIR}/libcharmm.so"
  rm -rf "$CMAKE_BUILD_DIR" 2>/dev/null || true
  echo "Installed tier lib: ${LIB_DIR}/libcharmm.so"
}

if [[ "$NEED_BUILD" == 1 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "(dry-run) would rebuild CHARMM into ${TIER_DIR}"
  else
    # Snakemake launches many cells at once; without a lock, parallel cmake runs and
    # api_func.F90 patches corrupt each other on NFS home.
    (
      flock -x 200
      if [[ -f "${LIB_DIR}/libcharmm.so" ]] && [[ ! "$PATCH_DST" -nt "${LIB_DIR}/libcharmm.so" ]]; then
        echo "Reusing tier build (another job finished while waiting): ${LIB_DIR}/libcharmm.so"
      else
        _run_tier_build
      fi
    ) 200>"$LOCK_FILE"
  fi
else
  echo "Reusing existing tier build: ${LIB_DIR}/libcharmm.so"
fi

export CHARMM_LIB_DIR="$LIB_DIR"
echo "export CHARMM_LIB_DIR=${CHARMM_LIB_DIR}"
