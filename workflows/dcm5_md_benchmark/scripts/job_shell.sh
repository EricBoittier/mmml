#!/usr/bin/env bash
# Run one DCM:5 benchmark job (called from Snakemake).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
JOB_ID="${1:?usage: job_shell.sh JOB_ID}"

cd "$REPO_ROOT"

PY="${KARML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

# Prefer karml console script on PATH / venv (python -m karml does not work).
export KARML_BIN="${KARML_BIN:-}"
if [[ -z "$KARML_BIN" && -x "$REPO_ROOT/.venv/bin/karml" ]]; then
  export KARML_BIN="$REPO_ROOT/.venv/bin/karml"
fi

echo "=== job_shell: ${JOB_ID} ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "KARML_BIN=${KARML_BIN:-<unset>}" >&2
echo "KARML_CKPT=${KARML_CKPT:-<unset>}" >&2

# Fail fast with a clear message in stdout.log (Snakemake redirects stderr too).
"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from benchmark_lib import load_config, resolve_checkpoint

cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK:', cfg['checkpoint'], flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" "$JOB_ID" --config "$WORKFLOW_ROOT/config.yaml"
