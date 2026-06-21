#!/usr/bin/env bash
# Add .max_npr + tier api_func.F90 metadata to prebuilt tier dirs (no cmake rebuild).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
BUILD_ROOT="${CHARMM_BUILD_DIR:-$HOME/.cache/mmml-charmm-build}"

# Representative n_ml (monomers×5) that selects each PBC tier via ensure_charmm_mlpot_limits.sh
declare -A TIER_NML=(
  [4000000]=385
  [8000000]=2200
  [12000000]=3000
  [36000000]=1330
  [56000000]=2000
)

for cap in 4000000 8000000 12000000 36000000 56000000; do
  lib="${BUILD_ROOT}/tier_${cap}_nodomdec/lib/libcharmm.so"
  if [[ ! -f "$lib" ]]; then
    echo "skip tier_${cap}: ${lib} missing"
    continue
  fi
  n_ml="${TIER_NML[$cap]}"
  echo "=== tier_${cap}_nodomdec (n_ml=${n_ml}) ==="
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$n_ml" --pbc
  grep -E 'max_Npr' "${BUILD_ROOT}/tier_${cap}_nodomdec/api_func.F90" | head -1
  echo -n ".max_npr="
  cat "${BUILD_ROOT}/tier_${cap}_nodomdec/.max_npr"
  echo
done

echo "Done."
