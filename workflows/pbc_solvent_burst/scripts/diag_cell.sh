#!/usr/bin/env bash
# Quick post-mortem for one burst matrix cell (run on cluster after a Slurm failure).
# Usage: bash scripts/diag_cell.sh RUN_TAG
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: diag_cell.sh RUN_TAG}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"

exec "${KARML_PYTHON}" "$WORKFLOW_ROOT/scripts/diag_cell.py" "$RUN_TAG" \
  --config "$WORKFLOW_ROOT/config.yaml"
