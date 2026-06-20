#!/usr/bin/env bash
# Launch pbc_solvent_burst on Slurm via Snakemake executor plugin.
#
# Usage: snakemake_slurm.sh [MAX_JOBS]
#   MAX_JOBS defaults to tier pools from config.yaml (fast + slow when tiering on).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_launch_jobs, slurm_resources_cli
cfg = load_config()
print(slurm_launch_jobs(cfg))
print(slurm_resources_cli(cfg))
")
EOF

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

echo "Snakemake Slurm: -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
