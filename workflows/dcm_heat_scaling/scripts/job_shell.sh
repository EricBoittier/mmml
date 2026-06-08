#!/usr/bin/env bash
# Run one DCM:N heat scaling job (called from Snakemake).
# Usage: job_shell.sh N_MONOMERS REPEAT DT_SLUG
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
N_MONOMERS="${1:?usage: job_shell.sh N_MONOMERS REPEAT DT_SLUG}"
REPEAT="${2:?usage: job_shell.sh N_MONOMERS REPEAT DT_SLUG}"
DT_SLUG="${3:?usage: job_shell.sh N_MONOMERS REPEAT DT_SLUG}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

echo "=== dcm_heat_scaling: DCM:${N_MONOMERS} repeat=${REPEAT} ${DT_SLUG} ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "MMML_BIN=${MMML_BIN:-<python -m mmml.cli.__main__>}" >&2
echo "MMML_CKPT=${MMML_CKPT:-<unset>}" >&2
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}" >&2

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from heat_lib import load_config, resolve_checkpoint

cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK:', cfg['checkpoint'], flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" "$N_MONOMERS" "$REPEAT" "$DT_SLUG" \
  --config "$WORKFLOW_ROOT/config.yaml"
