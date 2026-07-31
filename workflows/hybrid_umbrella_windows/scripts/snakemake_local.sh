#!/usr/bin/env bash
# Launch hybrid_umbrella_windows locally (interactive GPU node or dry-run).
# Usage: bash scripts/snakemake_local.sh [MAX_JOBS] [extra snakemake args...]
#
#   MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
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
export CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-}"
export MMML_CKPT="${MMML_CKPT:-}"
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:-}"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:-}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-}"
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
  --resources gpu="${JOBS}" charmm_slot="${JOBS}" \
  --keep-going \
  "$@"
