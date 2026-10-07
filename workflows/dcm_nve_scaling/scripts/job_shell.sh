#!/usr/bin/env bash
# Run one DCM:N NVE scaling job (called from Snakemake).
# Usage: job_shell.sh N_MONOMERS [INBFRQ]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
N_MONOMERS="${1:?usage: job_shell.sh N_MONOMERS [INBFRQ]}"
INBFRQ="${2:-}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

echo "=== dcm_nve_scaling: DCM:${N_MONOMERS} inbfrq=${INBFRQ:-config} ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "KARML_BIN=${KARML_BIN:-<python -m karml.cli.__main__>}" >&2
echo "KARML_CKPT=${KARML_CKPT:-<unset>}" >&2

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from scaling_lib import load_config, resolve_checkpoint

cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK:', cfg['checkpoint'], flush=True)
"

ARGS=("$WORKFLOW_ROOT/scripts/run_job.py" "$N_MONOMERS" --config "$WORKFLOW_ROOT/config.yaml")
if [[ -n "$INBFRQ" ]]; then
  ARGS+=(--inbfrq "$INBFRQ")
fi
exec "$PY" "${ARGS[@]}"
