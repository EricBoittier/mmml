#!/usr/bin/env bash
# Reap orphaned mmml MPI rank-0 workers left after Ctrl+C, prterun exit, or crashes.
#
# Orphans typically show PPID=1 (reparented init) while still holding CPU/GPU.
#
# Usage:
#   ./scripts/mmml-kill-orphans.sh              # list candidates (dry-run)
#   ./scripts/mmml-kill-orphans.sh --kill       # SIGTERM, then SIGKILL after 5s
#   ./scripts/mmml-kill-orphans.sh --kill --all # all mmml.cli workers, not only orphans
#   ./scripts/mmml-kill-orphans.sh --kill --keep 350967
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=1
KILL_ALL=0
QUIET=0
KILL_WAIT_SECS=5
declare -a KEEP_PIDS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

List or kill orphaned mmml rank-0 Python workers (mmml.cli / md-system under mpirun).

Options:
  --kill, -k       Send signals (default: dry-run listing only)
  --all            Kill every mmml.cli worker for \$USER, not only orphans (PPID=1)
  --keep PID       Do not kill this PID (repeatable)
  --quiet, -q      Suppress non-error output unless something would be killed
  -h, --help       Show this help

Examples:
  $ROOT/scripts/mmml-kill-orphans.sh
  $ROOT/scripts/mmml-kill-orphans.sh --kill
  $ROOT/scripts/mmml-kill-orphans.sh --kill --keep "\$(pgrep -f 'mmml.cli.__main__' | head -1)"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill | -k)
      DRY_RUN=0
      ;;
    --all)
      KILL_ALL=1
      ;;
    --keep)
      KEEP_PIDS+=("${2:?--keep requires PID}")
      shift
      ;;
    --quiet | -q)
      QUIET=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "$(basename "$0"): unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

is_kept() {
  local pid=$1
  local k
  for k in "${KEEP_PIDS[@]}"; do
    [[ "$pid" == "$k" ]] && return 0
  done
  return 1
}

mmml_worker_pids() {
  local pid cmd
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    [[ -n "$cmd" ]] || continue
    if [[ "$cmd" == *"mmml.cli.__main__"* ]] \
      || [[ "$cmd" == *"mmml md-system"* ]] \
      || [[ "$cmd" == *"/mmml/.venv/bin/python"* && "$cmd" == *"mmml"* && "$cmd" == *"md-system"* ]]; then
      printf '%s\n' "$pid"
    fi
  done < <(pgrep -u "$USER" -f 'mmml\.cli\.__main__|mmml md-system' 2>/dev/null || true)
}

is_orphan_worker() {
  local pid=$1
  local ppid pname
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  [[ -n "$ppid" ]] || return 1
  if [[ "$ppid" == "1" ]]; then
    return 0
  fi
  pname=$(ps -o comm= -p "$ppid" 2>/dev/null || true)
  # prterun exited but python rank survived as child of init-like stub
  if [[ "$pname" == "prterun" || "$pname" == "mpirun" ]]; then
    if ! kill -0 "$ppid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

declare -a TARGET_PIDS=()

while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  is_kept "$pid" && continue
  if [[ "$KILL_ALL" == 1 ]] || is_orphan_worker "$pid"; then
    TARGET_PIDS+=("$pid")
  fi
done < <(mmml_worker_pids | sort -u)

if [[ ${#TARGET_PIDS[@]} -eq 0 ]]; then
  [[ "$QUIET" == 0 ]] && echo "mmml-kill-orphans: no matching workers"
  exit 0
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "mmml-kill-orphans: would kill ${#TARGET_PIDS[@]} worker(s) (use --kill):"
  for pid in "${TARGET_PIDS[@]}"; do
    ps -o pid=,ppid=,pcpu=,etime=,args= -p "$pid" 2>/dev/null || true
  done
  exit 0
fi

echo "mmml-kill-orphans: stopping ${#TARGET_PIDS[@]} worker(s)..." >&2
for pid in "${TARGET_PIDS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

deadline=$((SECONDS + KILL_WAIT_SECS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  alive=0
  for pid in "${TARGET_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [[ "$alive" == 0 ]] && break
  sleep 0.5
done

for pid in "${TARGET_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "mmml-kill-orphans: SIGKILL pid=$pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

for pid in "${TARGET_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "mmml-kill-orphans: warning: pid=$pid still alive" >&2
    exit 1
  fi
done

[[ "$QUIET" == 0 ]] && echo "mmml-kill-orphans: done"
exit 0
