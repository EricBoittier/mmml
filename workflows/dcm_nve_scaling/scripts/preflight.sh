#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${KARML_CKPT:-}" ]]; then
  echo "KARML_CKPT is not set. Export your DCM PhysNet checkpoint directory." >&2
  exit 1
fi

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

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

echo "KARML_CKPT=${KARML_CKPT}"
echo "Per-step output: dcd_nsavc=dyn_nprint=nprint=1"
