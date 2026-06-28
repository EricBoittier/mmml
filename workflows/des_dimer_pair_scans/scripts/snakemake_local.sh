#!/usr/bin/env bash
# Launch des_dimer_pair_scans locally.
# Usage: bash scripts/snakemake_local.sh [MAX_JOBS]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/local}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
else
  CFG_PATH="$WORKFLOW_ROOT/$_cfg_raw"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
CONFIG_ARGS=()
if [[ "$CFG_PATH" != "$WORKFLOW_ROOT/config.yaml" ]]; then
  CONFIG_ARGS=(--configfile "$CFG_PATH")
fi

JOBS="${1:-2}"
shift || true

echo "Snakemake local: config=${CFG_PATH} -j${JOBS}" >&2
exec uv run --with snakemake snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources cpu=4 \
  --keep-going \
  "$@"
