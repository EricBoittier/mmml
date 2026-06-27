#!/usr/bin/env bash
# Step 1 for pc-bach: validate all CHARMM tiers required by the workflow matrix.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${MMML_WORKFLOW_CONFIG:-config.pc-bach.cpu.yaml}"
if [[ "${1:-}" == "--config" ]]; then
  CFG="${2:?--config requires path}"
fi

export MMML_CLUSTER="${MMML_CLUSTER:-pc-bach}"
# shellcheck source=../../../scripts/pc_bach_env.sh
source "$REPO_ROOT/scripts/pc_bach_env.sh"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

echo "=== pc-bach Step 1: tier lib checks (config=${CFG}) ==="

mapfile -t ROWS < <(
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

cfg = load_config(Path('${WORKFLOW_ROOT}') / '${CFG}')
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
    print(n_ml, box)
"
)

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "ERROR: no matrix cells in ${CFG}" >&2
  exit 1
fi

failed=0
for row in "${ROWS[@]}"; do
  read -r n_ml box <<<"$row"
  echo "--- n_ml=${n_ml} L=${box} ---"
  if ! bash "$REPO_ROOT/scripts/check_charmm_tier_lib.sh" --n-ml "$n_ml" --pbc --box-size "$box"; then
    failed=1
  fi
done

if [[ "$failed" == 1 ]]; then
  echo "" >&2
  echo "Step 1 FAILED: build missing tiers on pc-bach (source pc_bach_env.sh first), then re-run this script." >&2
  echo "  bash scripts/prebuild_charmm_tiers.sh --config ${CFG}" >&2
  exit 1
fi

echo "Step 1 OK: all required tier libs valid."
