#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${MMML_CKPT:-}" ]]; then
  echo "MMML_CKPT is not set. Export your DCM PhysNet checkpoint directory." >&2
  exit 1
fi

PY="${MMML_PYTHON:-python3}"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi

"$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from scaling_lib import load_config, resolve_checkpoint, _assert_per_step_output
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
_assert_per_step_output(cfg)
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK')
"

echo "MMML_CKPT=${MMML_CKPT}"
echo "Per-step output: dcd_nsavc=dyn_nprint=nprint=1"
