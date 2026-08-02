#!/usr/bin/env bash
# On agent stop: if dense_dt campaign unfinished, request a follow-up turn.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MONITOR="${ROOT}/scripts/slurm/dense_dt_campaign/monitor_and_progress.sh"

if [[ -x "$MONITOR" ]]; then
  bash "$MONITOR" >/dev/null 2>&1 || true
fi

done_n=0
if compgen -G "${ROOT}/artifacts/lj_scales/dense_dt_campaign/*/bench.log" >/dev/null; then
  done_n=$(rg -c '^RESULT ' "${ROOT}/artifacts/lj_scales/dense_dt_campaign"/*/bench.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
fi
n_queue=$(squeue -u "${USER:-boittier}" -h -o '%j' 2>/dev/null | grep -c '^ddc-' || true)
box24=0
box26=0
[[ -f "${ROOT}/artifacts/lj_scales/liquid_dense_L24/box.json" ]] && box24=1
[[ -f "${ROOT}/artifacts/lj_scales/liquid_dense_L26/box.json" ]] && box26=1

if [[ "${done_n:-0}" -ge 8 ]]; then
  printf '%s\n' '{}'
  exit 0
fi

python3 - "$done_n" "$n_queue" "$box24" "$box26" <<'PY'
import json, sys
done_n, n_queue, box24, box26 = sys.argv[1:5]
msg = (
    "Continue the denser-box / dt / ensemble overnight campaign for the "
    "condensed-phase hybrid ML/MM manuscript (§7 conservation, §8 DCM liquid density).\n\n"
    "Read artifacts/lj_scales/dense_dt_campaign/STATUS.md and monitor.log. "
    f"Snapshot: RESULT={done_n}, ddc_queue={n_queue}, L24_ready={box24}, L26_ready={box26}.\n\n"
    "Act: run monitor --react if needed, fix stuck box builds / failed Slurm jobs, "
    "keep MD progressing, and when H5s exist compare E_tot / H_NHC / bond health vs sparse L30. "
    "Do not stop while jobs or boxes are unfinished."
)
print(json.dumps({"followup_message": msg}))
PY
