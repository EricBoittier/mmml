#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

CONFIG="${1:-$WORKFLOW_ROOT/config.prep_sweep.yaml}"
exec "$PY" "$WORKFLOW_ROOT/scripts/collect_prep_sweep.py" --config "$CONFIG"
