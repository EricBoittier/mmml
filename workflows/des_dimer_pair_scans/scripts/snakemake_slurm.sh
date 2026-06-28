#!/usr/bin/env bash
# Launch des_dimer_pair_scans on Slurm (CPU nodes; ORCA/xTB/CHARMM).
# Usage: bash scripts/snakemake_slurm.sh [MAX_JOBS]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
elif [[ "$_cfg_raw" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$_cfg_raw")" && pwd)/$(basename "$_cfg_raw")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
CONFIG_ARGS=()
if [[ "$CFG_PATH" != "$WORKFLOW_ROOT/config.yaml" ]]; then
  CONFIG_ARGS=(--configfile "$CFG_PATH")
fi

JOBS="${1:-8}"
shift || true
PARTITION="${SLURM_PARTITION:-cpu}"
RES="cpu=${JOBS} mem_mb=$((JOBS * 8000))"

echo "Snakemake Slurm: config=${CFG_PATH} partition=${PARTITION} -j${JOBS}" >&2
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${RES} \
  --set-default-resources "slurm_partition=${PARTITION}" \
  --keep-going \
  "$@"
