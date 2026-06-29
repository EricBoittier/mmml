#!/usr/bin/env bash
# Append workflow + Slurm status every INTERVAL seconds (default 3600 = 1 hr).
#
# Usage:
#   cd workflows/pbc_liquid_density_dyn
#   MMML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml \
#     nohup bash scripts/hourly_status_watch.sh >> results/hourly_status.log 2>&1 &
#
#   tail -f results/hourly_status.log
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$WORKFLOW_ROOT"

INTERVAL="${HOURLY_STATUS_INTERVAL_SEC:-3600}"
CFG="${MMML_WORKFLOW_CONFIG:-config.pc-bach.cpu.yaml}"
if [[ "$CFG" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${CFG}"
fi

LOG="${HOURLY_STATUS_LOG:-$WORKFLOW_ROOT/results/hourly_status.log}"
mkdir -p "$(dirname "$LOG")"

_stamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

_snapshot() {
  {
    echo ""
    echo "================================================================"
    echo "[$(_stamp)] host=$(hostname) workflow=$(basename "$CFG_PATH")"
    echo "================================================================"
    echo "--- squeue -u ${USER:-?} ---"
    squeue -u "${USER:-$(whoami)}" -o '%.10i %.12P %.24j %.2t %.10M %R' 2>&1 || true
    echo ""
    echo "--- campaign status ---"
    if [[ -f "$WORKFLOW_ROOT/scripts/status.sh" ]]; then
      MMML_WORKFLOW_CONFIG="$CFG" bash "$WORKFLOW_ROOT/scripts/status.sh" 2>&1 || true
    fi
    echo ""
    echo "--- snakemake driver (last 8 lines) ---"
    if [[ -f "$WORKFLOW_ROOT/snakemake_slurm.log" ]]; then
      tail -8 "$WORKFLOW_ROOT/snakemake_slurm.log" 2>&1 || true
    else
      echo "(no snakemake_slurm.log)"
    fi
    echo ""
    echo "--- MPI spatial NP sweep ---"
    if [[ -f "$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep/results.csv" ]]; then
      column -t -s, "$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep/results.csv" 2>/dev/null \
        || cat "$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep/results.csv"
    elif [[ -f "$REPO_ROOT/artifacts/mpi_cpu_np_sweep/results.csv" ]]; then
      column -t -s, "$REPO_ROOT/artifacts/mpi_cpu_np_sweep/results.csv" 2>/dev/null \
        || cat "$REPO_ROOT/artifacts/mpi_cpu_np_sweep/results.csv"
    else
      echo "(no mpi_spatial_cpu_np_sweep/results.csv yet)"
    fi
    if [[ -f "$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep/sweep.log" ]]; then
      echo "spatial sweep tail:"
      tail -3 "$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep/sweep.log" 2>&1 || true
    elif [[ -f "$REPO_ROOT/artifacts/mpi_cpu_np_sweep/sweep.log" ]]; then
      echo "sweep tail:"
      tail -3 "$REPO_ROOT/artifacts/mpi_cpu_np_sweep/sweep.log" 2>&1 || true
    fi
    echo ""
    echo "--- done.txt count ---"
    out_root="$(python3 -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('${CFG_PATH}').read_text())
print(cfg.get('output_root', 'artifacts/pbc_liquid_density_dyn'))
" 2>/dev/null || echo artifacts/pbc_liquid_density_dyn_pc_bach)"
    art="$REPO_ROOT/$out_root"
    if [[ -d "$art" ]]; then
      find "$art" -name done.txt 2>/dev/null | wc -l | xargs -I{} echo "done.txt: {}/$(find "$art" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) cells"
    fi
  } >>"$LOG"
  echo "[$(_stamp)] snapshot -> $LOG"
}

echo "[$(_stamp)] hourly_status_watch start interval=${INTERVAL}s config=${CFG_PATH} log=${LOG}"
_snapshot
while true; do
  sleep "$INTERVAL"
  _snapshot
done
