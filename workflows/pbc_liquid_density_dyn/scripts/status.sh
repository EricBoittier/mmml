#!/usr/bin/env bash
# Campaign health dashboard (delegates to collect_diagnostics matrix).
#
# Usage:
#   bash scripts/status.sh
#   KARML_WORKFLOW_CONFIG=config.gpu08.local.yaml bash scripts/status.sh -v
#   bash scripts/status.sh --config config.yaml --tag dcm_277_t300_l32
#   bash scripts/status.sh --plot-dir results/plots --json results/status.json
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${KARML_WORKFLOW_CONFIG:-config.yaml}"
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

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"

exec "${KARML_PYTHON}" "$WORKFLOW_ROOT/scripts/collect_diagnostics.py" \
  --config "$CFG_PATH" \
  matrix \
  "${EXTRA_ARGS[@]}"
