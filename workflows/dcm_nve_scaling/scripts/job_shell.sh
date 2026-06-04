#!/usr/bin/env bash
# Run one DCM:N NVE scaling job (called from Snakemake).
# Usage: job_shell.sh N_MONOMERS [INBFRQ]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
N_MONOMERS="${1:?usage: job_shell.sh N_MONOMERS [INBFRQ]}"
INBFRQ="${2:-}"

cd "$REPO_ROOT"

PY="${MMML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

export MMML_BIN="${MMML_BIN:-}"
if [[ -z "$MMML_BIN" && -x "$REPO_ROOT/.venv/bin/mmml" ]]; then
  export MMML_BIN="$REPO_ROOT/.venv/bin/mmml"
fi

echo "=== dcm_nve_scaling: DCM:${N_MONOMERS} inbfrq=${INBFRQ:-config} ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "MMML_CKPT=${MMML_CKPT:-<unset>}" >&2

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
