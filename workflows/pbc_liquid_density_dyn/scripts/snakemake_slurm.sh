#!/usr/bin/env bash
# Launch pbc_liquid_density_dyn on Slurm.
# Usage: snakemake_slurm.sh [MAX_JOBS]
#   MMML_SNAKEMAKE_PROFILE=profiles/slurm-cpu MMML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml ...
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm}"
CFG="${MMML_WORKFLOW_CONFIG:-config.yaml}"
CONFIG_ARGS=()
if [[ "$CFG" != "config.yaml" ]]; then
  CONFIG_ARGS=(--configfile "$CFG")
fi

IFS=$'\t' read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_launch_jobs, slurm_resources_cli
cfg = load_config(Path('${CFG}'))
print(f\"{slurm_launch_jobs(cfg)}\t{slurm_resources_cli(cfg)}\")
")
EOF

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

echo "Snakemake Slurm: profile=${PROFILE} config=${CFG} -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
