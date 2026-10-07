#!/usr/bin/env bash
# Preflight before Snakemake: KARML_CKPT must exist and not be a README placeholder.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"

PY="${KARML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
PY="${PY:-python3}"

if [[ -z "${KARML_CKPT:-}" ]]; then
  echo "ERROR: KARML_CKPT is not set." >&2
  echo "  export KARML_CKPT=\$HOME/.../ckpts/dcm1-..." >&2
  exit 1
fi

exec "$PY" -c "
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from benchmark_lib import validate_checkpoint
from pathlib import Path
p = Path('${KARML_CKPT}').expanduser().resolve()
validate_checkpoint(p)
print(f'KARML_CKPT OK: {p}')
"
