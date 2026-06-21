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
    select_npr_tier,
    tier_max_npr,
)

cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
by_cap: dict[int, int] = {}
for cell in iter_matrix_cells(cfg):
    n_ml = estimate_ml_atoms(cell.n_monomers)
    cap = tier_max_npr(select_npr_tier(n_ml, pbc=True))
    by_cap[cap] = max(by_cap.get(cap, 0), n_ml)
for cap in sorted(by_cap):
    print(by_cap[cap], cap)
"
)

if [[ ${#N_ML_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no matrix cells found" >&2
  exit 1
fi

for row in "${N_ML_LIST[@]}"; do
  n_ml="${row%% *}"
  cap="${row##* }"
  echo "=== ensure tier max_Npr=${cap} (n_ml=${n_ml}) ==="
  eval "$("$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$n_ml" --pbc | grep '^export ')"
  echo "CHARMM_LIB_DIR=${CHARMM_LIB_DIR}"
done

echo "Done. Tier libs under \${CHARMM_BUILD_DIR:-\$HOME/.cache/mmml-charmm-build}/tier_*_nodomdec/"
