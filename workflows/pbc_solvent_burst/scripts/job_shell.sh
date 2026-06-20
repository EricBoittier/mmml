#!/usr/bin/env bash
# Run one PBC solvent burst campaign (called from Snakemake).
# Usage: job_shell.sh SOLVENT N_MONOMERS
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
SOLVENT="${1:?usage: job_shell.sh SOLVENT N_MONOMERS}"
N_MONOMERS="${2:?usage: job_shell.sh SOLVENT N_MONOMERS}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

echo "=== pbc_solvent_burst: ${SOLVENT}:${N_MONOMERS} box=32 ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, resolve_checkpoint
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK:', cfg['checkpoint'], flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" "$SOLVENT" "$N_MONOMERS" \
  --config "$WORKFLOW_ROOT/config.yaml"
