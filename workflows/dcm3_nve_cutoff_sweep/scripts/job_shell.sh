#!/usr/bin/env bash
# Run one DCM:3 NVE cutoff sweep job (Snakemake helper).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
PRESET_ID="${1:?usage: job_shell.sh PRESET_ID GEOM_ID}"
GEOM_ID="${2:?usage: job_shell.sh PRESET_ID GEOM_ID}"

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

echo "=== cutoff sweep: preset=${PRESET_ID} geom=${GEOM_ID} ===" >&2
echo "MMML_CKPT=${MMML_CKPT:-<unset>}" >&2

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from cutoff_lib import load_config, resolve_checkpoint
resolve_checkpoint(str(load_config(Path('${WORKFLOW_ROOT}/config.yaml'))['checkpoint']))
print('Preflight OK', flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" "$PRESET_ID" "$GEOM_ID" \
  --config "$WORKFLOW_ROOT/config.yaml"
