#!/usr/bin/env bash
# Build all CHARMM MLpot tiers needed by the burst matrix (run once before Snakemake).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

echo "Scanning matrix for distinct PBC CHARMM tiers..."
mapfile -t N_ML_LIST < <(
  "$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import iter_matrix_cells, load_config
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
    estimate_ml_atoms,
    select_npr_tier_for_build,
    tier_max_npr,
)

cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
by_cap: dict[int, tuple[int, float | None]] = {}
for cell in iter_matrix_cells(cfg):
    n_ml = estimate_ml_atoms(cell.n_monomers, solvent=cell.solvent)
    cap = tier_max_npr(
        select_npr_tier_for_build(n_ml, pbc=True, box_side_A=cell.box_size)
    )
    prev = by_cap.get(cap)
    if prev is None or n_ml > prev[0]:
        by_cap[cap] = (n_ml, float(cell.box_size))
for cap in sorted(by_cap):
    n_ml, box = by_cap[cap]
    print(n_ml, cap, box)
"
)

if [[ ${#N_ML_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no matrix cells found" >&2
  exit 1
fi

for row in "${N_ML_LIST[@]}"; do
  read -r n_ml cap box <<<"$row"
  echo "=== ensure tier max_Npr=${cap} (n_ml=${n_ml} L=${box}) ==="
  eval "$(
    "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" \
      --n-ml "$n_ml" --pbc --box-size "$box" | grep '^export '
  )"
  echo "CHARMM_LIB_DIR=${CHARMM_LIB_DIR}"
done

echo "Done. Tier libs under \${CHARMM_BUILD_DIR:-\$HOME/.cache/mmml-charmm-build}/tier_*_nodomdec/"
echo "Each tier has lib/libcharmm.so, api_func.F90, and .max_npr stamp — jobs reuse these without rebuilding."
