#!/usr/bin/env bash
# Remote control for monitor_tv.py (tmux bindings + CLI).
# Usage: bash scripts/monitor_tv_ctl.sh next|prev|pause
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
ACTION="${1:?usage: monitor_tv_ctl.sh next|prev|pause}"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

TV="$WORKFLOW_ROOT/scripts/monitor_tv.py"
msg="$("$MMML_PYTHON" "$TV" ctl "$ACTION" --message 2>/dev/null || true)"

if [[ -n "${TMUX:-}" && -n "$msg" ]]; then
  tmux display-message -d 2000 "📺 $msg" 2>/dev/null || true
fi

if [[ -n "$msg" ]]; then
  echo "$msg"
fi
