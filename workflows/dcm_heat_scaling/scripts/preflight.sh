#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${MMML_CKPT:-}" ]]; then
  echo "MMML_CKPT is not set. Export your DCM PhysNet checkpoint directory." >&2
  exit 1
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from heat_lib import load_config, resolve_checkpoint
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
print('Preflight OK')
"

echo "MMML_CKPT=${MMML_CKPT}"
"$PY" -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('${WORKFLOW_ROOT}/config.yaml').read_text())
print('cluster_sizes:', cfg['cluster_sizes'])
print('repeats:', cfg.get('repeats', [1]))
print('dt_fs_values:', cfg['dt_fs_values'])
"
