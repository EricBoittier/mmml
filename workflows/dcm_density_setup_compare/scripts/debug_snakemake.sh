#!/usr/bin/env bash
# Snakemake driver + Slurm log triage (pc-studix login node).
#
# Usage (from ~/mmml/workflows/dcm_density_setup_compare):
#   bash scripts/debug_snakemake.sh
#   bash scripts/debug_snakemake.sh --job 203595
#   bash scripts/debug_snakemake.sh --tag resilient_dcm_154_t50_l32_ht_hoover
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"
debug_bootstrap_cluster

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
SLURM_LOG_DIR="$(debug_slurm_log_dir)"

echo "=== dcm_density_setup_compare Snakemake debug ==="
echo "host:     $(hostname)"
echo "workflow: $WORKFLOW_ROOT"
echo "user:     ${USER:-?}"
echo

echo "=== Your Slurm queue ==="
debug_user_gpu_queue || true
echo

if pgrep -af 'snakemake --profile profiles/slurm' >/dev/null 2>&1; then
  echo "=== Running Snakemake driver ==="
  pgrep -af 'snakemake --profile profiles/slurm' || true
  echo
else
  echo "=== Snakemake driver ==="
  echo "(not running — start with: nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &)"
  echo
fi

if [[ -f "$DRIVER_LOG" ]]; then
  debug_grep_section "snakemake_slurm.log (errors / restarts)" "$DRIVER_LOG" "$DBG_PAT_SLURM" 40
  echo "=== Recent failed external_jobids (last 10) ==="
  grep -oE 'external_jobid: [0-9]+' "$DRIVER_LOG" 2>/dev/null | tail -10 || echo "(none)"
  echo
else
  echo "=== snakemake_slurm.log ==="
  echo "(missing: $DRIVER_LOG)"
  echo
fi

if [[ -n "$SLURM_JOB" ]]; then
  echo "=== sacct $SLURM_JOB ==="
  debug_slurm_job_summary "$SLURM_JOB"
  echo
  slurm_log="$(debug_find_slurm_log "$SLURM_JOB")"
  if [[ -n "$slurm_log" ]]; then
    echo "log: $slurm_log"
    debug_grep_section "Slurm rule log" "$slurm_log" \
      "$DBG_PAT_SLURM|error|Error|Traceback|libOpenCL|MMML_CKPT" 60
  else
    echo "(no rule log under $SLURM_LOG_DIR for job $SLURM_JOB)"
  fi
  echo
fi

if [[ -n "$TAG" ]]; then
  exec bash "$WORKFLOW_ROOT/scripts/debug_cell.sh" "$TAG" ${SLURM_JOB:+--job "$SLURM_JOB"}
fi

if [[ -d "$SLURM_LOG_DIR" && -z "$SLURM_JOB" ]]; then
  echo "=== Recent Slurm rule logs (newest 5 by mtime) ==="
  mapfile -t recent_logs < <(
    find "$SLURM_LOG_DIR" -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -5 | cut -d' ' -f2-
  )
  if [[ ${#recent_logs[@]} -eq 0 ]]; then
    echo "(no logs under $SLURM_LOG_DIR)"
  else
    for f in "${recent_logs[@]}"; do
      job_base="$(basename "$f" .log)"
      echo "--- ${job_base} ($(basename "$(dirname "$f")")) ---"
      debug_slurm_job_summary "$job_base" 2>/dev/null | sed 's/^/  /' || true
      grep -nE "$DBG_PAT_SLURM|error|Error|Traceback|libOpenCL" "$f" 2>/dev/null | tail -12 \
        | sed 's/^/  /' || echo "  (clean or empty)"
      echo
    done
  fi
fi
