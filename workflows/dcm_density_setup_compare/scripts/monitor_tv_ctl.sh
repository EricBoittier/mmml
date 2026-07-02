#!/usr/bin/env bash
# Remote control for monitor_tv.py (bound to tmux keys in monitor_tmux.sh).
# Usage: bash scripts/monitor_tv_ctl.sh next|prev|pause
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
ACTION="${1:?usage: monitor_tv_ctl.sh next|prev|pause}"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

exec "$MMML_PYTHON" "$WORKFLOW_ROOT/scripts/monitor_tv.py" ctl "$ACTION"
