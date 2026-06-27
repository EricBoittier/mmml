#!/usr/bin/env bash
# Step 1: validate a CHARMM MLpot tier lib before launching jobs (skip rebuild when OK).
#
# Usage:
#   source scripts/pc_bach_env.sh   # pc-bach OpenMPI stack
#   bash scripts/check_charmm_tier_lib.sh --n-ml 2660 --pbc --box-size 32
#
# Exit 0 when the selected tier lib exists, stamp matches, and mmml mpi-check passes.
# Exit 1 when a rebuild is required or mpi-check fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"

N_ML=""
PBC=0
BOX_SIZE=""
SOURCE_PC_BACH=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --n-ml N_ML_ATOMS [--pbc] [--box-size L_ANGSTROM] [--pc-bach]

Validates ensure_charmm_mlpot_limits tier selection without building.
When the tier lib is present and mpi-check OK, prints "tier lib valid" and exits 0.
Otherwise prints what is missing and exits 1 (run ensure_charmm_mlpot_limits or prebuild).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-ml) N_ML="${2:?--n-ml requires value}"; shift 2 ;;
    --pbc) PBC=1; shift ;;
    --box-size) BOX_SIZE="${2:?--box-size requires value}"; shift 2 ;;
    --pc-bach) SOURCE_PC_BACH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$N_ML" ]]; then
  echo "ERROR: --n-ml is required" >&2
  usage >&2
  exit 2
fi

if [[ "$SOURCE_PC_BACH" == 1 || "${MMML_CLUSTER:-}" == "pc-bach" ]]; then
  # shellcheck source=pc_bach_env.sh
  source "$ROOT/scripts/pc_bach_env.sh"
fi

mmml_resolve_env "$ROOT"
PY="${MMML_PYTHON}"

PBC_FLAG=()
[[ "$PBC" == 1 ]] && PBC_FLAG=(--pbc)
BOX_FLAG=()
[[ -n "$BOX_SIZE" ]] && BOX_FLAG=(--box-size "$BOX_SIZE")

echo "=== check_charmm_tier_lib (dry-run tier selection) ==="
DRY_OUT="$("$ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" "${PBC_FLAG[@]}" "${BOX_FLAG[@]}" --dry-run)"
echo "$DRY_OUT"

eval "$(echo "$DRY_OUT" | grep '^export ' || true)"
if [[ -z "${CHARMM_LIB_DIR:-}" ]]; then
  echo "ERROR: could not resolve CHARMM_LIB_DIR from dry-run" >&2
  exit 1
fi

LIB="${CHARMM_LIB_DIR}/libcharmm.so"
STAMP="${CHARMM_LIB_DIR%/lib}/.max_npr"

if [[ ! -f "$LIB" ]]; then
  echo "MISSING: $LIB — run ensure_charmm_mlpot_limits.sh (or prebuild_charmm_tiers.sh) after sourcing pc_bach_env.sh" >&2
  exit 1
fi

if [[ ! -f "$STAMP" ]]; then
  echo "WARN: tier lib exists but missing stamp $STAMP (legacy prebuild); ensure_charmm_mlpot_limits will adopt on first job" >&2
else
  echo "tier stamp: $(cat "$STAMP") ($STAMP)"
fi

echo "libcharmm.so: $LIB ($(du -h "$LIB" | awk '{print $1}'))"
if command -v ldd >/dev/null 2>&1; then
  if ldd "$LIB" 2>/dev/null | grep -q 'libmpi'; then
    echo "MPI link: yes ($(ldd "$LIB" 2>/dev/null | grep libmpi | head -1 | awk '{print $1, $3}'))"
  else
    echo "ERROR: $LIB is not MPI-linked (wrong build?)" >&2
    exit 1
  fi
fi

echo "=== mmml mpi-check ==="
export CHARMM_LIB_DIR
if ! "$PY" -m mmml.cli.__main__ mpi-check; then
  echo "ERROR: mmml mpi-check failed for CHARMM_LIB_DIR=$CHARMM_LIB_DIR" >&2
  exit 1
fi

echo "tier lib valid: $LIB"
exit 0
