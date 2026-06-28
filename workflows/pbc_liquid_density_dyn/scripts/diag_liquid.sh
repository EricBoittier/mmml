#!/usr/bin/env bash
# Liquid-density diagnostics CLI (matrix, cell, collect, trajectory).
#
# Usage:
#   bash scripts/diag_liquid.sh matrix
#   bash scripts/diag_liquid.sh matrix --config config.gpu08.local.yaml -v
#   bash scripts/diag_liquid.sh collect --config config.yaml --output-dir results/diagnostics
#   bash scripts/diag_liquid.sh cell dcm_277_t300_l32 --config config.yaml
#   bash scripts/diag_liquid.sh trajectory dcm_277_t300_l32 --config config.yaml
#
# Shorthand (production config):
#   MMML_WORKFLOW_CONFIG=config.yaml bash scripts/diag_liquid.sh matrix -v --plot-dir results/plots
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${MMML_WORKFLOW_CONFIG:-config.yaml}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CFG="${2:?--config requires path}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$CFG" = /* ]]; then
  CFG_PATH="$CFG"
elif [[ "$CFG" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${CFG}"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

exec "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/collect_diagnostics.py" \
  --config "$CFG_PATH" \
  "${EXTRA_ARGS[@]}"
