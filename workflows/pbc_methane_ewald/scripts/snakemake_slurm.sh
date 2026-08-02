#!/usr/bin/env bash
# Launch pbc_methane_ewald on Slurm via Snakemake executor plugin.
#
# Usage: snakemake_slurm.sh [MAX_JOBS]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

CFG="${MMML_WORKFLOW_CONFIG:-config.yaml}"

IFS=$'\t' read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_launch_jobs, slurm_resources_cli
cfg = load_config(Path('${CFG}') if Path('${CFG}').is_absolute() else Path('${WORKFLOW_ROOT}') / '${CFG}')
print(f\"{slurm_launch_jobs(cfg)}\t{slurm_resources_cli(cfg)}\")
")
EOF

if [[ -z "${DEFAULT_RES:-}" ]]; then
  echo "snakemake_slurm.sh: failed to resolve --resources from config" >&2
  exit 1
fi

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm}"
export MMML_WORKFLOW_CONFIG="$CFG"

exec "$PY" -m snakemake \
  --profile "$PROFILE" \
  --jobs "$JOBS" \
  --resources $DEFAULT_RES \
  --keep-going \
  "$@"
