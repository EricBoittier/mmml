#!/usr/bin/env bash
# Stop all Snakemake drivers for this workflow (uv wrapper + python child).
# Usage: bash scripts/stop_snakemake.sh
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_pids_for_workflow() {
  local pid cwd
  for pid in $(pgrep -f 'snakemake --profile profiles/slurm|uv run --with snakemake' 2>/dev/null || true); do
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    if [[ "$cwd" == "$WORKFLOW_ROOT" ]]; then
      echo "$pid"
    fi
  done | sort -u
}

mapfile -t PIDS < <(_pids_for_workflow)
if ((${#PIDS[@]} == 0)); then
  echo "No snakemake drivers in ${WORKFLOW_ROOT}"
  exit 0
fi

echo "Stopping ${#PIDS[@]} driver process(es) in ${WORKFLOW_ROOT}:"
for pid in "${PIDS[@]}"; do
  ps -p "$pid" -o pid=,args= 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
done

sleep 2
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "SIGKILL ${pid}"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

mapfile -t LEFT < <(_pids_for_workflow)
if ((${#LEFT[@]} > 0)); then
  echo "WARNING: still running: ${LEFT[*]}" >&2
  exit 1
fi
echo "Done. Run: snakemake --profile profiles/slurm --unlock"
