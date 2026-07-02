#!/usr/bin/env bash
# Snakemake driver and Slurm log triage for dcm_density_setup_compare.
#
# Usage:
#   bash scripts/debug_snakemake.sh
#   bash scripts/debug_snakemake.sh --job 203595
#   bash scripts/debug_snakemake.sh --tag resilient_dcm_154_t50_l32_ht_hoover
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"

SLURM_JOB=""
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      SLURM_JOB="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

DRIVER_LOG="$WORKFLOW_ROOT/snakemake_slurm.log"
SLURM_LOG_DIR="$WORKFLOW_ROOT/.snakemake/slurm_logs/rule_run_setup_compare"

echo "=== dcm_density_setup_compare Snakemake debug ==="
echo "workflow: $WORKFLOW_ROOT"
echo

if pgrep -af 'snakemake --profile profiles/slurm' >/dev/null 2>&1; then
  echo "=== Running Snakemake driver ==="
  pgrep -af 'snakemake --profile profiles/slurm' || true
  echo
else
  echo "=== Snakemake driver ==="
  echo "(not running)"
  echo
fi

if [[ -f "$DRIVER_LOG" ]]; then
  debug_grep_section "snakemake_slurm.log (errors / restarts)" "$DRIVER_LOG" "$DBG_PAT_SLURM" 40
else
  echo "=== snakemake_slurm.log ==="
  echo "(missing: $DRIVER_LOG)"
  echo
fi

if [[ -n "$SLURM_JOB" ]]; then
  echo "=== Slurm job $SLURM_JOB ==="
  found=0
  if [[ -d "$SLURM_LOG_DIR" ]]; then
    while IFS= read -r -d '' f; do
      found=1
      echo "log: $f"
      debug_grep_section "Slurm log" "$f" "$DBG_PAT_SLURM|error|Error|Traceback" 60
    done < <(find "$SLURM_LOG_DIR" -name "${SLURM_JOB}.log" -print0 2>/dev/null)
  fi
  if [[ "$found" -eq 0 ]]; then
    echo "(no log under $SLURM_LOG_DIR for job $SLURM_JOB)"
  fi
  echo
fi

if [[ -n "$TAG" ]]; then
  LOG="$(debug_cell_log "$TAG")"
  echo "=== Cell stdout for $TAG ==="
  if [[ -f "$LOG" ]]; then
    debug_grep_section "stdout.log" "$LOG" \
      "error:|watchdog|Post MLpot SD|Pre-dynamics GRMS|heat segment|overlap|post-overlap-rescue|Failed leg" 50
  else
    echo "(missing: $LOG)"
  fi
fi

if [[ -d "$SLURM_LOG_DIR" && -z "$SLURM_JOB" && -z "$TAG" ]]; then
  echo "=== Recent Slurm rule logs (newest 5) ==="
  mapfile -t recent_logs < <(ls -t "$SLURM_LOG_DIR"/*/*.log 2>/dev/null | head -5 || true)
  if [[ ${#recent_logs[@]} -eq 0 ]]; then
    echo "(no logs under $SLURM_LOG_DIR)"
  else
    for f in "${recent_logs[@]}"; do
      echo "--- $(basename "$f") ---"
      grep -nE "$DBG_PAT_SLURM|error|Error|Traceback" "$f" 2>/dev/null | tail -15 || echo "(clean or empty)"
      echo
    done
  fi
fi
