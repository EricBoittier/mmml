#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from bulk_density import bulk_reference_table, matrix_uses_bulk_density
from campaign_lib import (
    load_config,
    matrix_box_sizes,
    matrix_job_count,
    matrix_temperatures,
    resolve_checkpoint,
    slurm_launch_jobs,
    slurm_small_cluster_max_n,
    slurm_tier_enabled,
    slurm_tier_resource_pools,
    total_jaxmd_ps,
    total_pycharmm_equi_ps,
)
from cleanup_strategy import resolve_cleanup_strategy
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
ckpt = resolve_checkpoint(str(cfg['checkpoint']))
strategy = resolve_cleanup_strategy(cfg)
print('Preflight OK')
print('checkpoint:', ckpt)
print('cleanup_strategy:', strategy.name)
print('matrix jobs:', matrix_job_count(cfg))
print('temperatures:', matrix_temperatures(cfg))
print('box_sizes:', matrix_box_sizes(cfg))
if matrix_uses_bulk_density(cfg):
    print('bulk_density_fractions:', cfg.get('bulk_density_fractions'))
    print('bulk N at L (298 K liquid, monomers per box):')
    print(bulk_reference_table(matrix_box_sizes(cfg)))
print('solvents:', cfg['solvents'])
if matrix_uses_bulk_density(cfg):
    print('matrix mode: bulk-density fractions')
else:
    print('cluster_sizes:', cfg['cluster_sizes'])
print('jaxmd total ps:', total_jaxmd_ps(cfg))
print('pycharmm equi total ps:', total_pycharmm_equi_ps(cfg))
print('slurm tiering:', slurm_tier_enabled(cfg))
if slurm_tier_enabled(cfg):
    print('slurm_small_cluster_max_n:', slurm_small_cluster_max_n(cfg))
    print('slurm resource pools:', slurm_tier_resource_pools(cfg))
print('slurm launch -j:', slurm_launch_jobs(cfg))
if matrix_uses_bulk_density(cfg):
    print('NOTE: bulk-density matrix targets configured fractions of liquid N (see N_bulk table).')
else:
    print('NOTE: large cluster_sizes in a small box are very dense; expect overlap rescue at large N.')
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import (
    NPR_TIERS,
    estimate_ml_atoms,
    select_npr_tier,
    validate_mlpot_system_size,
)
from campaign_lib import iter_matrix_cells
worst = max(
    (
        (
            estimate_ml_atoms(c.n_monomers, solvent=c.solvent),
            c.box_size,
        )
        for c in iter_matrix_cells(cfg)
    ),
    key=lambda item: select_npr_tier(
        item[0], pbc=True, box_side_A=item[1]
    ),
)
max_n_ml, worst_box = worst
tier = select_npr_tier(max_n_ml, pbc=True, box_side_A=worst_box)
print(f'max matrix n_ml={max_n_ml} L={worst_box:g} CHARMM tier={tier} (max_Npr={NPR_TIERS[tier]}, PBC pairs)')
validate_mlpot_system_size(max_n_ml, pbc=True, box_side_A=worst_box)
"

if ! command -v packmol >/dev/null 2>&1; then
  echo "WARNING: packmol not on PATH (required for initial placement)." >&2
fi

if [[ -n "${MMML_CKPT:-}" ]]; then
  echo "MMML_CKPT=${MMML_CKPT} (optional override when config uses \${MMML_CKPT})"
fi
