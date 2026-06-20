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
from campaign_lib import (
    load_config,
    matrix_box_sizes,
    matrix_job_count,
    matrix_temperatures,
    resolve_checkpoint,
    slurm_max_concurrent,
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
print('solvents:', cfg['solvents'])
print('cluster_sizes:', cfg['cluster_sizes'])
print('jaxmd total ps:', total_jaxmd_ps(cfg))
print('pycharmm equi total ps:', total_pycharmm_equi_ps(cfg))
print('slurm_max_concurrent:', slurm_max_concurrent(cfg))
"

if ! command -v packmol >/dev/null 2>&1; then
  echo "WARNING: packmol not on PATH (required for initial placement)." >&2
fi

if [[ -n "${MMML_CKPT:-}" ]]; then
  echo "MMML_CKPT=${MMML_CKPT} (optional override when config uses \${MMML_CKPT})"
fi

max_n=100
box=32
echo "NOTE: ${max_n} monomers in ${box} Å cube is very dense; expect overlap rescue at large N."
