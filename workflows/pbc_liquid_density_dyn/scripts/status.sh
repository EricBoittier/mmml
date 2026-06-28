#!/usr/bin/env bash
# Status for liquid-density dynamics campaigns.
# Usage:
#   bash scripts/status.sh
#   MMML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml bash scripts/status.sh
#   bash scripts/status.sh --config config.pc-bach.cpu.yaml [--tag TAG] [--failed]
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

if [[ "$CFG" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${CFG}"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

exec "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/status.py" \
  --config "$CFG_PATH" \
  "${EXTRA_ARGS[@]}"
