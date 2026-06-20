#!/usr/bin/env bash
# Run one PBC solvent burst campaign (called from Snakemake).
# Usage: job_shell.sh RUN_TAG
#   e.g. job_shell.sh dcm_10_t300_l32
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: job_shell.sh RUN_TAG (e.g. dcm_10_t300_l32)}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

echo "=== pbc_solvent_burst: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, resolve_checkpoint, cell_from_tag
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print('Preflight OK:', cfg['checkpoint'], cell, flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" --tag "$RUN_TAG" \
  --config "$WORKFLOW_ROOT/config.yaml"
