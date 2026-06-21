#!/usr/bin/env bash
# Ensure libcharmm.so max_Npr tier fits the ML atom count for this job.
# Usage: ensure_charmm_mlpot_limits.sh --n-ml 2660 [--pbc]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
MMML_PY="$(mmml_resolve_python "$ROOT")"
N_ML=""
PBC=0
BOX_SIZE=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --n-ml N_ML_ATOMS [--pbc] [--box-size L_ANGSTROM] [--dry-run]

Selects the smallest local CHARMM build tier (default/large/xlarge/xxlarge/xxxlarge), builds
a tier-local api_func.F90 with the matching max_Npr, and installs:
  \${CHARMM_BUILD_DIR:-\$HOME/.cache/mmml-charmm-build}/tier_\${MAX_NPR}_nodomdec/lib/libcharmm.so

Tier libs are built with rebuild_charmm_mlpot.sh --no-domdec (MPI MLpot, np=1).
Pre-build all matrix tiers once:
  bash workflows/pbc_solvent_burst/scripts/prebuild_charmm_tiers.sh

Jobs only rebuild when that tier's lib is missing or stale — not when another tier
patches the repo api_func.F90.

Use --pbc for periodic systems (``mlpot_update`` image ML pairs).

Exports CHARMM_LIB_DIR to the tier lib directory for child processes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-ml) N_ML="${2:?--n-ml requires value}"; shift 2 ;;
    --pbc) PBC=1; shift ;;
    --box-size) BOX_SIZE="${2:?--box-size requires value}"; shift 2 ;;
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

read -r TIER TARGET TEMPLATE_F90 <<<"$(
  "$MMML_PY" - "$N_ML" "$ROOT" "$PBC" "$BOX_SIZE" <<'PY'
import sys
from pathlib import Path

n_ml = int(sys.argv[1])
root = Path(sys.argv[2])
pbc = bool(int(sys.argv[3]))
box_raw = sys.argv[4].strip() if len(sys.argv) > 4 else ""
box = float(box_raw) if box_raw else None
sys.path.insert(0, str(root))
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
    charmm_mlpot_limits_from_source,
    select_npr_tier_for_build,
    tier_max_npr,
)

tier = select_npr_tier_for_build(n_ml, pbc=pbc, box_side_A=box)
target = tier_max_npr(tier)
parsed = charmm_mlpot_limits_from_source()
template = parsed[2] if parsed else root / "setup" / "api" / "api_func.F90"
print(tier, target, template)
PY
)"

if [[ -z "${TIER:-}" || -z "${TARGET:-}" ]]; then
  echo "ERROR: could not resolve CHARMM NPR tier for N_ML=${N_ML}" >&2
  exit 1
fi

echo "ML atoms=${N_ML} pbc=${PBC} box=${BOX_SIZE:-<unset>} -> tier=${TIER} max_Npr=${TARGET}"

if [[ ! -f "$TEMPLATE_F90" ]]; then
  TEMPLATE_F90="$ROOT/setup/api/api_func.F90"
fi
if [[ ! -f "$TEMPLATE_F90" ]]; then
  echo "ERROR: api_func.F90 template not found (tried ${TEMPLATE_F90})" >&2
  exit 1
fi

BUILD_ROOT="${CHARMM_BUILD_DIR:-$HOME/.cache/mmml-charmm-build}"
TIER_DIR="${BUILD_ROOT}/tier_${TARGET}_nodomdec"
LIB_DIR="${TIER_DIR}/lib"
TIER_F90="${TIER_DIR}/api_func.F90"
STAMP_FILE="${TIER_DIR}/.max_npr"
LOCK_FILE="${BUILD_ROOT}/.tier_${TARGET}_nodomdec.build.lock"
mkdir -p "$LIB_DIR" "$BUILD_ROOT"

_sync_tier_metadata_mtime() {
  if [[ -f "${LIB_DIR}/libcharmm.so" ]]; then
    touch -r "${LIB_DIR}/libcharmm.so" "$TIER_F90" "$STAMP_FILE" 2>/dev/null || true
  fi
}

_write_tier_f90() {
  cp -f "$TEMPLATE_F90" "$TIER_F90"
  "$MMML_PY" - "$TIER_F90" "$TARGET" <<'PY'
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
  echo "$TARGET" >"$STAMP_FILE"
}

_stamp_matches() {
  [[ -f "$STAMP_FILE" ]] && [[ "$(cat "$STAMP_FILE")" == "$TARGET" ]]
}

# Older ensure_charmm_mlpot_limits.sh left libcharmm.so but no tier metadata.
_adopt_legacy_prebuild_if_needed() {
  if [[ -f "${LIB_DIR}/libcharmm.so" ]] && [[ ! -f "$STAMP_FILE" ]]; then
    echo "Adopting legacy tier prebuild (adding .max_npr stamp): ${LIB_DIR}/libcharmm.so"
    _write_tier_f90
    _sync_tier_metadata_mtime
  fi
}

_adopt_legacy_prebuild_if_needed

NEED_BUILD=0
if [[ ! -f "${LIB_DIR}/libcharmm.so" ]]; then
  NEED_BUILD=1
elif ! _stamp_matches; then
  NEED_BUILD=1
elif [[ ! -f "$TIER_F90" ]]; then
  _write_tier_f90
  _sync_tier_metadata_mtime
fi

_run_tier_build() {
  _write_tier_f90
  CMAKE_BUILD_DIR="${CHARMM_CMAKE_BUILD_DIR:-${SLURM_TMPDIR:-/tmp}/mmml-charmm-cmake-${TARGET}-$$}"
  echo "Building CHARMM MLpot tier ${TIER} (max_Npr=${TARGET}); cmake in ${CMAKE_BUILD_DIR}"
  rm -rf "$CMAKE_BUILD_DIR" 2>/dev/null || true
  mkdir -p "$CMAKE_BUILD_DIR"
  MMML_PATCH_SOURCE="$TIER_F90" \
    CHARMM_BUILD_DIR="$CMAKE_BUILD_DIR" \
    "$ROOT/scripts/rebuild_charmm_mlpot.sh" --clean --no-domdec
  mkdir -p "$LIB_DIR"
  cp -f "${CHARMM_HOME:-$ROOT/setup/charmm}/libcharmm.so" "${LIB_DIR}/libcharmm.so"
  _sync_tier_metadata_mtime
  rm -rf "$CMAKE_BUILD_DIR" 2>/dev/null || true
  echo "Installed tier lib: ${LIB_DIR}/libcharmm.so (max_Npr=${TARGET})"
}

if [[ "$NEED_BUILD" == 1 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "(dry-run) would build tier ${TIER} into ${TIER_DIR}"
  else
    # Snakemake launches many cells at once; flock so only one cmake per tier runs.
    (
      flock -x 200
      if [[ -f "${LIB_DIR}/libcharmm.so" ]] && _stamp_matches; then
        echo "Reusing tier build (another job finished while waiting): ${LIB_DIR}/libcharmm.so"
      else
        _run_tier_build
      fi
    ) 200>"$LOCK_FILE"
  fi
else
  echo "Reusing existing tier build: ${LIB_DIR}/libcharmm.so (max_Npr=${TARGET})"
fi

export CHARMM_LIB_DIR="$LIB_DIR"
echo "export CHARMM_LIB_DIR=${CHARMM_LIB_DIR}"
