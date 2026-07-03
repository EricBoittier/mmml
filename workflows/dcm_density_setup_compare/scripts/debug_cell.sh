#!/usr/bin/env bash
# Triage one matrix cell (pc-studix login node).
#
# Usage (from ~/mmml/workflows/dcm_density_setup_compare):
#   bash scripts/debug_cell.sh
#   bash scripts/debug_cell.sh resilient_dcm_52_t50_l28_ht_hoover
#   bash scripts/debug_cell.sh TAG --tail 50
#   bash scripts/debug_cell.sh TAG --job 203595   # also show Slurm sacct + rule log
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"
debug_bootstrap_cluster

DEFAULT_RUN_TAG="${MMML_DEFAULT_RUN_TAG:-resilient_dcm_52_t50_l28_ht_bussi_sw_ovlp25}"
TAG="${1:-$DEFAULT_RUN_TAG}"
shift || true

TAIL_ONLY=false
TAIL_N=50
SLURM_JOB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tail)
      TAIL_ONLY=true
      TAIL_N="${2:-50}"
      shift 2
      ;;
    --job)
      SLURM_JOB="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

ART="$(debug_cell_dir "$TAG")"
LOG="$(debug_cell_log "$TAG")"
SUMMARY="$ART/campaign_summary.json"
CAMPAIGN="$ART/campaign.yaml"

echo "=== dcm_density_setup_compare debug: $TAG ==="
echo "host:         $(hostname)"
echo "workflow:     $WORKFLOW_ROOT"
echo "artifact_dir: $ART"
echo "stdout.log:   $LOG"
echo "done:         $(debug_cell_done "$TAG")"
echo

if [[ ! -f "$LOG" ]]; then
  echo "ERROR: missing $LOG" >&2
  echo "Hint: job may still be queued — try: bash scripts/debug_snakemake.sh" >&2
  exit 1
fi

if $TAIL_ONLY; then
  grep -nE \
    "error:|watchdog|Post MLpot SD|Pre-dynamics GRMS|heat segment|overlap|post-overlap-rescue|Failed leg" \
    "$LOG" | tail -n "$TAIL_N"
  exit 0
fi

abort="$(debug_extract_abort "$LOG")"
mini_grms="$(debug_extract_mini_grms "$LOG")"
echo "abort:     ${abort:-<none yet>}"
echo "mini GRMS: ${mini_grms:-<not found>}"
echo

if [[ -n "$SLURM_JOB" ]]; then
  echo "=== sacct $SLURM_JOB ==="
  debug_slurm_job_summary "$SLURM_JOB"
  slurm_log="$(debug_find_slurm_log "$SLURM_JOB")"
  if [[ -n "$slurm_log" ]]; then
    debug_grep_section "Slurm rule log ($SLURM_JOB)" "$slurm_log" \
      "$DBG_PAT_SLURM|error|Error|Traceback|libOpenCL" 40
  fi
fi

debug_grep_section "Abort / hard errors" "$LOG" \
  "$DBG_PAT_ABORT|error:|ERROR|failed with exit code" 30

debug_grep_section "GRMS gates & thresholds" "$LOG" "$DBG_PAT_GRMS" 40

debug_grep_section "Mini / SD watchdog" "$LOG" "$DBG_PAT_MINI" 40

debug_grep_section "Heat / overlap / rescue" "$LOG" "$DBG_PAT_HEAT" 60

debug_grep_section "Campaign legs" "$LOG" "$DBG_PAT_LEGS" 30

debug_grep_section "MPI / bonded BLOCK" "$LOG" "$DBG_PAT_MPI" 20

if [[ -f "$SUMMARY" ]]; then
  echo "=== campaign_summary.json (failed legs) ==="
  debug_print_campaign_summary "$SUMMARY" || grep -E 'job_id|exit_code|backend' "$SUMMARY" | head -40
  echo
else
  echo "=== campaign_summary.json ==="
  echo "(missing: $SUMMARY — job may have aborted before summary was written)"
  echo
fi

if [[ -f "$CAMPAIGN" ]]; then
  echo "=== campaign.yaml job order ==="
  grep -E '^  [a-z_0-9]+:' "$CAMPAIGN" | head -20 || true
  echo
fi

echo "=== stdout.log (last 30 lines) ==="
tail -30 "$LOG"
