#!/usr/bin/env bash
# Launch label_acquisition on Slurm.
# Usage: bash scripts/snakemake_slurm.sh [MAX_JOBS] [extra snakemake args...]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
else
  CFG_PATH="$WORKFLOW_ROOT/$_cfg_raw"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
CONFIG_ARGS=()
if [[ "$CFG_PATH" != "$WORKFLOW_ROOT/config.yaml" ]]; then
  CONFIG_ARGS=(--configfile "$CFG_PATH")
fi

JOBS="${1:-8}"
shift || true

echo "Snakemake slurm: config=${CFG_PATH} --jobs ${JOBS}" >&2
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  --jobs "$JOBS" \
  --keep-going \
  "$@"
