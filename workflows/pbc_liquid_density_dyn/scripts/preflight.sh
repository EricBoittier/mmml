#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "${1:-}" == "--config" ]]; then
  CFG="${2:?--config requires path}"
fi
if [[ "$CFG" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${CFG}"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
from pathlib import Path
import sys
_BURST = Path('${WORKFLOW_ROOT}').parent / 'pbc_solvent_burst' / 'scripts'
sys.path.insert(0, str(_BURST))
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from bulk_density import bulk_reference_table, matrix_uses_bulk_density
from campaign_lib import (
    campaign_job_order,
    load_config,
    matrix_box_sizes,
    matrix_job_count,
    matrix_temperatures,
    mlpot_device_name,
    resolve_checkpoint,
    scheduler_mode,
    slurm_launch_jobs,
    slurm_resources_cli,
    total_pycharmm_equi_ps,
    total_pycharmm_prod_ps,
)
from cleanup_strategy import resolve_cleanup_strategy
cfg_path = Path('${CFG_PATH}')
cfg = load_config(cfg_path)
ckpt = resolve_checkpoint(str(cfg['checkpoint']))
strategy = resolve_cleanup_strategy(cfg)
print('Preflight OK — pbc_liquid_density_dyn')
print('config:', cfg_path)
print('scheduler:', scheduler_mode(cfg), 'mlpot_device:', mlpot_device_name(cfg))
print('checkpoint:', ckpt)
print('cleanup_strategy:', strategy.name)
print('matrix jobs:', matrix_job_count(cfg))
print('campaign legs:', len(campaign_job_order(cfg)), campaign_job_order(cfg)[:4], '…')
print('temperatures:', matrix_temperatures(cfg))
print('box_sizes:', matrix_box_sizes(cfg))
if matrix_uses_bulk_density(cfg):
    print('bulk_density_fractions:', cfg.get('bulk_density_fractions'))
    print('bulk N at L (298 K liquid):')
    print(bulk_reference_table(matrix_box_sizes(cfg)))
print('pycharmm equi total ps:', total_pycharmm_equi_ps(cfg))
print('pycharmm prod total ps:', total_pycharmm_prod_ps(cfg))
print('warmup_mlpot_jax:', cfg.get('warmup_mlpot_jax', True))
print('mpirun_wrapper:', cfg.get('mpirun_wrapper'))
print('slurm partition:', cfg.get('slurm_partition'))
print('slurm launch -j:', slurm_launch_jobs(cfg))
print('slurm resources:', slurm_resources_cli(cfg))
"
