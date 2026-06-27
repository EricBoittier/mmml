#!/usr/bin/env bash
# Status for liquid-density dynamics campaigns.
# Usage: bash scripts/status.sh [--tag TAG] [--failed]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

exec "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/status.py" \
  --config "$WORKFLOW_ROOT/config.yaml" \
  "$@"
