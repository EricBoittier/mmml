#!/usr/bin/env bash
# tmux dashboard: driver log, Slurm queue, Snakemake triage, optional cell stdout.
#
# Usage (from ~/mmml/workflows/dcm_density_setup_compare):
#   bash scripts/monitor_tmux.sh
#   bash scripts/monitor_tmux.sh --tag resilient_dcm_52_t50_l28_ht_bussi
#   bash scripts/monitor_tmux.sh --log snakemake_prep_sweep.log --replace
#   bash scripts/monitor_tmux.sh --session prep --log snakemake_prep_sweep.log
#
# Layout (2×2):
#   driver log (tail)     | cell stdout or driver pgrep
#   squeue watch          | debug_snakemake watch
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"

SESSION="dcm"
DRIVER_LOG=""
TAG=""
REPLACE=false
ATTACH=true

usage() {
  sed -n '2,12p' "$0"
  echo
  echo "Options:"
  echo "  --session NAME   tmux session name (default: dcm)"
  echo "  --log FILE       driver log under workflow root (default: snakemake_slurm.log)"
  echo "  --tag RUN_TAG    tail artifacts/.../RUN_TAG/stdout.log in top-right pane"
  echo "  --replace        kill existing session before creating layout"
  echo "  --no-attach      create session and exit (attach later: tmux attach -t NAME)"
  echo "  -h, --help       show this help"
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
    --tag)
      TAG="$2"
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

if [[ -z "$DRIVER_LOG" ]]; then
  case "${MMML_WORKFLOW_CONFIG:-config.yaml}" in
    *prep_sweep*) DRIVER_LOG="snakemake_prep_sweep.log" ;;
    *) DRIVER_LOG="snakemake_slurm.log" ;;
  esac
fi
if [[ "$DRIVER_LOG" != /* ]]; then
  DRIVER_LOG="$WORKFLOW_ROOT/$DRIVER_LOG"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if $REPLACE; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session '$SESSION' already exists — attaching (use --replace to recreate layout)"
    exec tmux attach-session -t "$SESSION"
  fi
fi

CELL_LOG=""
if [[ -n "$TAG" ]]; then
  CELL_LOG="$(debug_cell_log "$TAG")"
fi

# 2×2 grid: pane 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
tmux new-session -d -s "$SESSION" -c "$WORKFLOW_ROOT" -n monitor

tmux split-window -h -t "$SESSION:0" -p 50
tmux select-pane -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION:0.0" -p 50
tmux select-pane -t "$SESSION:0.1"
tmux split-window -v -t "$SESSION:0.1" -p 50

tmux select-pane -t "$SESSION:0.0" -T "driver"
tmux send-keys -t "$SESSION:0.0" \
  "echo '=== driver log: ${DRIVER_LOG} ==='; tail -F '${DRIVER_LOG}'" C-m

tmux select-pane -t "$SESSION:0.1" -T "cell"
if [[ -n "$CELL_LOG" ]]; then
  tmux send-keys -t "$SESSION:0.1" \
    "echo '=== cell stdout: ${TAG} ==='; tail -F '${CELL_LOG}'" C-m
else
  tmux send-keys -t "$SESSION:0.1" \
    "watch -n 15 'echo \"=== Snakemake driver ===\"; pgrep -af \"snakemake --profile profiles/slurm\" 2>/dev/null || echo \"(not running — nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &)\"; echo; echo \"Tip: bash scripts/monitor_tmux.sh --replace --tag RUN_TAG\"'" C-m
fi

tmux select-pane -t "$SESSION:0.2" -T "queue"
tmux send-keys -t "$SESSION:0.2" \
  "watch -n 10 'squeue -u \"\${USER}\" -o \"%.10i %.9P %.12M %.8T %.6D %R %.40j\" 2>/dev/null | head -25'" C-m

tmux select-pane -t "$SESSION:0.3" -T "triage"
tmux send-keys -t "$SESSION:0.3" \
  "watch -n 30 'bash scripts/debug_snakemake.sh 2>/dev/null | tail -45'" C-m

tmux select-pane -t "$SESSION:0.0"

echo "Created tmux session '$SESSION' in $WORKFLOW_ROOT"
echo "  top-left:  tail -F $(basename "$DRIVER_LOG")"
if [[ -n "$TAG" ]]; then
  echo "  top-right: tail -F $TAG/stdout.log"
else
  echo "  top-right: driver pgrep (pass --tag RUN_TAG for cell log)"
fi
echo "  bottom:    squeue + debug_snakemake (refreshed)"
echo
echo "Detach: Ctrl-b d   Reattach: tmux attach -t $SESSION"

if $ATTACH; then
  exec tmux attach-session -t "$SESSION"
fi
