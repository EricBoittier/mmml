#!/usr/bin/env bash
# Run one DCM:5 benchmark job (called from Snakemake).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
JOB_ID="${1:?usage: job_shell.sh JOB_ID}"

cd "$REPO_ROOT"

PY="${MMML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" "$JOB_ID" --config "$WORKFLOW_ROOT/config.yaml"
