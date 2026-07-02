#!/usr/bin/env bash
# Triage one matrix cell from stdout.log and campaign artifacts.
#
# Usage:
#   bash scripts/debug_cell.sh RUN_TAG
#   bash scripts/debug_cell.sh resilient_dcm_52_t50_l28_ht_hoover
#   bash scripts/debug_cell.sh RUN_TAG --tail 50   # quick one-liner only
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=debug_lib.sh
source "$WORKFLOW_ROOT/scripts/debug_lib.sh"

TAG="${1:?usage: debug_cell.sh RUN_TAG [--tail N]}"
shift || true

TAIL_ONLY=false
TAIL_N=50
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tail)
      TAIL_ONLY=true
      TAIL_N="${2:-50}"
      shift 2
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
echo "artifact_dir: $ART"
echo "stdout.log:   $LOG"
echo "done:         $(debug_cell_done "$TAG")"
echo

if [[ ! -f "$LOG" ]]; then
  echo "ERROR: missing $LOG" >&2
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

debug_grep_section "Abort / hard errors" "$LOG" \
  "$DBG_PAT_ABORT|error:|ERROR|failed with exit code" 30

debug_grep_section "GRMS gates & thresholds" "$LOG" "$DBG_PAT_GRMS" 40

debug_grep_section "Mini / SD watchdog" "$LOG" "$DBG_PAT_MINI" 40

debug_grep_section "Heat / overlap / rescue" "$LOG" "$DBG_PAT_HEAT" 60

debug_grep_section "Campaign legs" "$LOG" "$DBG_PAT_LEGS" 30

debug_grep_section "MPI / bonded BLOCK" "$LOG" "$DBG_PAT_MPI" 20

if [[ -f "$SUMMARY" ]]; then
  echo "=== campaign_summary.json (failed legs) ==="
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import json
from pathlib import Path
p = Path("$SUMMARY")
data = json.loads(p.read_text())
jobs = data.get("jobs", data if isinstance(data, list) else [])
for j in jobs:
    rc = int(j.get("exit_code", 0))
    mark = "FAIL" if rc else "ok "
    print(f"  [{mark}] {j.get('job_id')} backend={j.get('backend')} exit_code={rc}")
failed = [j.get("job_id") for j in jobs if int(j.get("exit_code", 0)) != 0]
if failed:
    print("failed:", failed)
PY
  else
    grep -E 'job_id|exit_code|backend' "$SUMMARY" | head -40
  fi
  echo
else
  echo "=== campaign_summary.json ==="
  echo "(missing: $SUMMARY)"
  echo
fi

if [[ -f "$CAMPAIGN" ]]; then
  echo "=== campaign.yaml job order ==="
  grep -E '^  [a-z_0-9]+:' "$CAMPAIGN" | head -20 || true
  echo
fi

echo "=== stdout.log (last 30 lines) ==="
tail -30 "$LOG"
