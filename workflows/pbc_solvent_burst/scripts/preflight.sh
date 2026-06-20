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
    resolve_checkpoint,
    slurm_max_concurrent,
    total_jaxmd_ps,
    total_pycharmm_equi_ps,
)
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
ckpt = resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK')
print('checkpoint:', ckpt)
print('solvents:', cfg['solvents'])
print('cluster_sizes:', cfg['cluster_sizes'])
print('box_size:', cfg['box_size'])
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
