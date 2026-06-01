#!/usr/bin/env bash
# Rebuild libcharmm.so after changing MLpot limits in source/api/api_func.F90.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARMM_HOME="${CHARMM_HOME:-$ROOT/setup/charmm}"
BUILD_DIR="$CHARMM_HOME/build/cmake"
LIB_OUT="$CHARMM_HOME/libcharmm.so"

F90="$CHARMM_HOME/source/api/api_func.F90"
if [[ ! -f "$F90" ]]; then
  echo "rebuild_charmm_mlpot: missing $F90" >&2
  exit 1
fi

echo "MLpot limits in source:"
grep -E 'max_Nml|max_Npr' "$F90" || true

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "rebuild_charmm_mlpot: missing cmake build dir $BUILD_DIR" >&2
  echo "Configure CHARMM first (see setup/charmm/build/cmake)." >&2
  exit 1
fi

echo "Building libcharmm.so in $BUILD_DIR ..."
cmake --build "$BUILD_DIR" -j "$(nproc)"

BUILT="$(find "$CHARMM_HOME/build" -name 'libcharmm.so' -print -quit)"
if [[ -z "$BUILT" ]]; then
  echo "rebuild_charmm_mlpot: build finished but libcharmm.so not found under $CHARMM_HOME/build" >&2
  exit 1
fi

cp -f "$BUILT" "$LIB_OUT"
echo "Installed $LIB_OUT"
echo "Verify: python -c \"from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import mlpot_limits_message; print(mlpot_limits_message())\""
