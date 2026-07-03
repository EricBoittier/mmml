#!/usr/bin/env bash
# tmux "TV studio" for dcm_density_setup_compare: rotating job channels + driver log.
#
# Usage (from ~/mmml/workflows/dcm_density_setup_compare):
#   bash scripts/monitor_tmux.sh
#   bash scripts/monitor_tmux.sh --tags resilient_dcm_77_t50_l32_ht_bussi resilient_dcm_52_t50_l28_ht_bussi
#   bash scripts/monitor_tmux.sh --interval 8 --include-done --replace
#   bash scripts/monitor_tmux.sh --log snakemake_prep_sweep.log --session prep
#
# Layout:
#   left (38%):  Snakemake driver log (tail -F)
#   right (62%): Rich TV dashboard — auto-rotates matrix cells like channels
#
# Keys (Ctrl-b then):
#   n       next channel
#   p       previous channel
#   Space   pause / resume auto-rotate
#   d       detach
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

SESSION="dcm-tv"
DRIVER_LOG=""
TAGS=()
REPLACE=false
ATTACH=true
INTERVAL=12
INCLUDE_DONE=false
CONFIG="${MMML_WORKFLOW_CONFIG:-config.yaml}"

usage() {
  sed -n '2,18p' "$0"
  echo
  echo "Options:"
  echo "  --session NAME      tmux session (default: dcm-tv)"
  echo "  --log FILE          driver log (default: snakemake_slurm.log or prep sweep log)"
  echo "  --tags TAG [TAG..]    explicit TV channels (default: auto-discover matrix)"
  echo "  --interval SEC      auto-rotate interval (default: 12)"
  echo "  --include-done      keep finished cells in rotation"
  echo "  --config FILE       workflow config for channel discovery"
  echo "  --replace           kill existing session and recreate"
  echo "  --no-attach         create detached (tmux attach -t NAME)"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"
      shift 2
      ;;
    --log)
      DRIVER_LOG="$2"
      shift 2
      ;;
    --tags)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TAGS+=("$1")
        shift
      done
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --include-done)
      INCLUDE_DONE=true
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --replace)
      REPLACE=true
      shift
      ;;
    --no-attach)
      ATTACH=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found in PATH" >&2
  exit 1
fi

export MMML_WORKFLOW_CONFIG="$CONFIG"

if [[ -z "$DRIVER_LOG" ]]; then
  case "$CONFIG" in
    *prep_sweep*) DRIVER_LOG="snakemake_prep_sweep.log" ;;
    *) DRIVER_LOG="snakemake_slurm.log" ;;
  esac
fi
if [[ "$DRIVER_LOG" != /* ]]; then
  DRIVER_LOG="$WORKFLOW_ROOT/$DRIVER_LOG"
fi

CTL="$WORKFLOW_ROOT/scripts/monitor_tv_ctl.sh"
PY="$MMML_PYTHON"
TV="$WORKFLOW_ROOT/scripts/monitor_tv.py"

_apply_tv_bindings() {
  # User bindings (override tmux default n/p = next/prev *window*).
  tmux bind-key -T prefix n run-shell "bash '$CTL' next"
  tmux bind-key -T prefix p run-shell "bash '$CTL' prev"
  tmux bind-key -T prefix Space run-shell "bash '$CTL' pause"
  tmux bind-key -T prefix N run-shell "bash '$CTL' next"
  tmux bind-key -T prefix P run-shell "bash '$CTL' prev"
  tmux bind-key -T prefix Right run-shell "bash '$CTL' next"
  tmux bind-key -T prefix Left run-shell "bash '$CTL' prev"
}

TV_ARGS=(live --interval "$INTERVAL" --driver-log "$(basename "$DRIVER_LOG")")
if $INCLUDE_DONE; then
  TV_ARGS+=(--include-done)
fi
if [[ -n "$CONFIG" ]]; then
  TV_ARGS+=(--config "$CONFIG")
fi
if ((${#TAGS[@]} > 0)); then
  TV_ARGS+=(--tags "${TAGS[@]}")
fi

# Seed channel state for ctl + list
INIT_ARGS=(init --config "$CONFIG")
if $INCLUDE_DONE; then
  INIT_ARGS+=(--include-done)
fi
if ((${#TAGS[@]} > 0)); then
  INIT_ARGS+=(--tags "${TAGS[@]}")
fi
"$PY" "$TV" "${INIT_ARGS[@]}" >/dev/null 2>&1 || true

LAUNCH="$WORKFLOW_ROOT/.monitor_tv/launch.sh"
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
  printf 'export MMML_WORKFLOW_CONFIG=%q\n' "$CONFIG"
  printf 'exec %q ' "$PY" "$TV"
  printf '%q ' "${TV_ARGS[@]}"
  printf '\n'
} >"$LAUNCH"
chmod +x "$LAUNCH"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if $REPLACE; then
    tmux kill-session -t "$SESSION"
  else
    echo "Session '$SESSION' exists — refreshing key bindings (use --replace to rebuild panes)"
    _apply_tv_bindings
    exec tmux attach-session -t "$SESSION"
  fi
fi

mkdir -p "$WORKFLOW_ROOT/.monitor_tv"

tmux new-session -d -s "$SESSION" -c "$WORKFLOW_ROOT" -n "📺 TV"
tmux split-window -h -t "$SESSION:0" -p 62

tmux select-pane -t "$SESSION:0.0" -T "driver"
tmux send-keys -t "$SESSION:0.0" \
  "printf '%s\n' '╔══════════════════════════════════════╗' '║  SNAKEMAKE DRIVER LOG                ║' '╚══════════════════════════════════════╝' '' '  tail -F ${DRIVER_LOG}' '' ; tail -F '${DRIVER_LOG}'" C-m

tmux select-pane -t "$SESSION:0.1" -T "channels"
tmux send-keys -t "$SESSION:0.1" "exec '$LAUNCH'" C-m

# Studio look + on-screen key hints
tmux set-option -t "$SESSION" status-style 'bg=colour235,fg=colour141'
tmux set-option -t "$SESSION" status-left-length 40
tmux set-option -t "$SESSION" status-right-length 60
tmux set-option -t "$SESSION" status-left '#[bold fg=colour213] 📺 DCM-TV #[fg=colour81]|#[default] '
tmux set-option -t "$SESSION" status-right '#[dim]focus TV: n/p/Space · ^B n/p/←/→ · %H:%M#[default]'
tmux set-window-option -t "$SESSION:0" window-status-current-style 'bg=colour236,fg=colour213,bold'

_apply_tv_bindings

tmux select-pane -t "$SESSION:0.1"

echo "Created tmux TV session '$SESSION'"
echo "  left:  tail -F $(basename "$DRIVER_LOG")"
echo "  right: rotating channels (every ${INTERVAL}s)"
echo
echo "  Focus TV pane (right):  n / p / Space"
echo "  From any pane:          Ctrl-b n/p/Space or Ctrl-b ←/→"
echo "  Ctrl-b d                detach"
echo
echo "  Channel list: $PY scripts/monitor_tv.py list --config $CONFIG"
echo "  Reattach:     tmux attach -t $SESSION"

if $ATTACH; then
  exec tmux attach-session -t "$SESSION"
fi
