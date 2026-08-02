#!/usr/bin/env bash
# Watchdog for denser-box + dt/x64 campaign (manuscript §§7–8: conservation + liquid ρ).
# Usage:
#   bash scripts/slurm/dense_dt_campaign/monitor_and_progress.sh           # report
#   bash scripts/slurm/dense_dt_campaign/monitor_and_progress.sh --react   # report + remediate
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT=artifacts/lj_scales/dense_dt_campaign
LOG_DIR="${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/STATUS.md"
MONITOR_LOG="${OUT_ROOT}/monitor.log"
MARKER_BOX_BUILD=/tmp/build_dense_boxes_v3.sh
REACT=0
[[ "${1:-}" == "--react" ]] && REACT=1

mkdir -p "$OUT_ROOT" "$LOG_DIR"
ts="$(date -Is)"
host="$(hostname)"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

{
  echo "## dense_dt_campaign monitor $ts host=$host sha=$sha react=$REACT"
  echo
} | tee -a "$MONITOR_LOG"

box_ready() {
  local d="$1"
  [[ -f "${d}/box.json" ]] && { [[ -f "${d}/model.psf" && -f "${d}/model.crd" ]] || [[ -f "${d}/mini.psf" && -f "${d}/mini.crd" ]]; }
}

BOX24=artifacts/lj_scales/liquid_dense_L24
BOX26=artifacts/lj_scales/liquid_dense_L26
BOX30=artifacts/lj_scales/liquid_nvt

# --- Slurm ddc jobs ---
DDC_LINES=()
while IFS= read -r _ddc; do
  [[ -n "$_ddc" ]] && DDC_LINES+=("$_ddc")
done < <(squeue -u "$USER" -h -o '%i %j %T %M %R' 2>/dev/null | awk '/ddc-/ {print}' || true)
n_run=0; n_pend=0
for line in "${DDC_LINES[@]}"; do
  st=$(awk '{print $3}' <<<"$line")
  [[ "$st" == RUNNING ]] && n_run=$((n_run+1))
  [[ "$st" == PENDING ]] && n_pend=$((n_pend+1))
done

# Completed / failed from job_ids + sacct
declare -a NEED_RESUBMIT=()
if [[ -f "$OUT_ROOT/job_ids.txt" ]]; then
  while read -r _line; do
    # SUBMITTED <tag> -> job <jid> ...
    tag=$(awk '{print $2}' <<<"$_line")
    jid=$(awk '{for(i=1;i<=NF;i++) if($i=="job"){print $(i+1); exit}}' <<<"$_line")
    [[ -z "${tag:-}" || -z "${jid:-}" || ! "$jid" =~ ^[0-9]+$ ]] && continue
    state=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)
    if [[ "$state" == FAILED || "$state" == TIMEOUT || "$state" == NODE_FAIL || "$state" == CANCELLED ]]; then
      if [[ ! -f "$OUT_ROOT/${tag}/SUCCESS.flag" ]] && ! squeue -j "$jid" -h >/dev/null 2>&1; then
        # still in queue? skip
        if ! squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "ddc-${tag}"; then
          NEED_RESUBMIT+=("$tag")
        fi
      fi
    fi
    if [[ "$state" == COMPLETED ]]; then
      touch "$OUT_ROOT/${tag}/SUCCESS.flag" 2>/dev/null || true
    fi
  done < "$OUT_ROOT/job_ids.txt"
fi

# --- Box build process ---
box_build_alive=0
pgrep -f 'build_dense_boxes_v[23]\.sh|liquid-box .*liquid_dense_L' >/dev/null 2>&1 && box_build_alive=1
packmol_alive=0
pgrep -x packmol >/dev/null 2>&1 && packmol_alive=1

# Packmol stuck: >45 min with no model.crd
packmol_stuck=0
if [[ "$packmol_alive" -eq 1 ]] && ! box_ready "$BOX24"; then
  etime=$(ps -C packmol -o etimes= 2>/dev/null | awk 'NR==1{print $1+0}')
  if [[ -n "${etime:-}" && "$etime" -gt 2700 ]]; then
    packmol_stuck=1
  fi
fi

# --- Status body ---
{
  echo "# dense_dt_campaign STATUS"
  echo
  echo "Updated: \`$ts\` on \`$host\` (sha \`$sha\`)"
  echo
  echo "## Manuscript targets (keep in mind)"
  echo
  echo "- §7 Energy conservation / integrator robustness (NVE ΔE, NHC invariant)"
  echo "- §8 Pure liquids: density & structure — DCM hybrid NPT/NVT toward ρ≈1.33 g/cm³"
  echo "- Outside-loss observables: density, then ΔH_vap (see docs/npt-density-dh-dg.md)"
  echo "- This campaign: denser box + dt/x64 + NVT/NPT/NVE with PSF angle restraints"
  echo
  echo "## Boxes"
  echo
  for d in "$BOX24" "$BOX26" "$BOX30"; do
    if box_ready "$d"; then
      rho=$(python3 -c "import json; d=json.load(open('${d}/box.json')); print(d.get('density_g_cm3') or d.get('certified_density_g_cm3') or d)" 2>/dev/null || echo '?')
      echo "- READY \`$d\` (ρ info: $rho)"
    else
      echo "- NOT READY \`$d\`"
    fi
  done
  echo
  echo "- box_build_alive=$box_build_alive packmol_alive=$packmol_alive packmol_stuck=$packmol_stuck"
  echo
  echo "## Slurm ddc-*"
  echo
  echo "- running=$n_run pending=$n_pend"
  if ((${#DDC_LINES[@]} > 0)); then
    echo '```'
    printf '%s\n' "${DDC_LINES[@]}"
    echo '```'
  else
    echo "- (no ddc jobs in queue)"
  fi
  echo
  echo "## Arms"
  echo
  echo '| tag | RESULT | h5 | notes |'
  echo '|---|---|---|---|'
  for dir in "$OUT_ROOT"/*/; do
    [[ -d "$dir" ]] || continue
    tag=$(basename "$dir")
    [[ "$tag" == logs ]] && continue
    res=$(rg -N '^RESULT ' "$dir/bench.log" 2>/dev/null | tail -1 || true)
    nh5=$(find "$dir" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')
    fail=""
    [[ -f "$dir/FAIL.txt" ]] && fail="FAIL.txt"
    echo "| $tag | ${res:-—} | $nh5 | $fail |"
  done
  if ((${#NEED_RESUBMIT[@]} > 0)); then
    echo
    echo "## Need resubmit"
    echo
    printf -- '- %s\n' "${NEED_RESUBMIT[@]}"
  fi
  echo
  echo "## Remediations this tick"
  echo
} > "$STATUS"

actions=()

if [[ "$REACT" -eq 1 ]]; then
  # Restart stuck packmol / dead box build
  if [[ "$packmol_stuck" -eq 1 ]]; then
    actions+=("kill stuck packmol (>45min) and restart dense box build v3")
    pkill -f 'build_dense_boxes_v[23]\.sh' 2>/dev/null || true
    pkill -x packmol 2>/dev/null || true
    sleep 2
    # fall through to ensure_build
    box_build_alive=0
  fi

  if [[ "$box_build_alive" -eq 0 ]] && { ! box_ready "$BOX24" || ! box_ready "$BOX26"; }; then
    actions+=("start/restart dense box build (L24 then L26, packmol-tol=1.5)")
    cat > "$MARKER_BOX_BUILD" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
cd /mmhome/boittier/home/mmml
source examples/lj_scales/_env.sh
export JAX_PLATFORMS=cpu LJ_DEVICE=cpu
LOG=/tmp/build_dense_boxes_v3.log
echo "START $(date -Is)" | tee -a "$LOG"
build() {
  local L=$1 RHO=$2 OUT=$3
  mkdir -p "$OUT"
  if [[ -f "$OUT/box.json" && -f "$OUT/model.crd" && -f "$OUT/model.psf" ]]; then
    echo "SKIP $OUT already certified $(date -Is)" | tee -a "$LOG"
    return 0
  fi
  # Wipe incomplete packmol scratch so tolerance fix takes effect
  rm -rf "$OUT/packmol_repack" "$OUT/.packmol_cache"
  echo "=== L=$L rho_target=$RHO tol=1.5 -> $OUT $(date -Is) ===" | tee -a "$LOG" | tee "$OUT/build.log"
  uv run mmml liquid-box \
    --composition DCM:120 \
    --box-size "$L" \
    --target-density-g-cm3 "$RHO" \
    --density-certify-relative-tolerance 0.08 \
    --packmol-tolerance 1.5 \
    --rebuild-packmol \
    --output-dir "$OUT" \
    --temperature 300 \
    --quiet \
    >>"$OUT/build.log" 2>&1
  local rc=$?
  echo "rc=$rc $(date -Is)" | tee -a "$LOG" | tee -a "$OUT/build.log"
  [[ -f "$OUT/box.json" ]] && python3 -c "import json; print(json.dumps(json.load(open('$OUT/box.json')), indent=2)[:800])" | tee -a "$LOG"
  return $rc
}
# L=24: slightly below absolute fill (1.224) so Packmol can finish
build 24 1.15 artifacts/lj_scales/liquid_dense_L24
build 26 0.96 artifacts/lj_scales/liquid_dense_L26
echo "ALL BOXES DONE $(date -Is)" | tee -a "$LOG"
EOS
    chmod +x "$MARKER_BOX_BUILD"
    SESSION=dense-box-build
    tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$SESSION" 2>/dev/null && tmux -f /exec-daemon/tmux.portal.conf kill-session -t "$SESSION" 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$SESSION" -c "$ROOT" -- bash "$MARKER_BOX_BUILD"
  fi

  # Resubmit failed arms if boxes ready
  if ((${#NEED_RESUBMIT[@]} > 0)); then
    for tag in "${NEED_RESUBMIT[@]}"; do
      # parse original submit line
      line=$(rg -N "SUBMITTED ${tag} " "$OUT_ROOT/job_ids.txt" | tail -1 || true)
      [[ -z "$line" ]] && continue
      # SUBMITTED TAG -> job JID  ens=E box=B dt=D x64=X ps=P
      ens=$(sed -n 's/.*ens=\([^ ]*\).*/\1/p' <<<"$line")
      box=$(sed -n 's/.*box=\([^ ]*\).*/\1/p' <<<"$line")
      dt=$(sed -n 's/.*dt=\([^ ]*\).*/\1/p' <<<"$line")
      x64=$(sed -n 's/.*x64=\([^ ]*\).*/\1/p' <<<"$line")
      ps=$(sed -n 's/.*ps=\([^ ]*\).*/\1/p' <<<"$line")
      case "$box" in
        24) bdir=$BOX24 ;;
        26) bdir=$BOX26 ;;
        30) bdir=$BOX30 ;;
        *) continue ;;
      esac
      box_ready "$bdir" || continue
      seed=$((100 + RANDOM % 800))
      actions+=("resubmit $tag ens=$ens box=$box dt=$dt x64=$x64 ps=$ps")
      jid=$(sbatch --parsable \
        --job-name="ddc-${tag}" \
        --output="${LOG_DIR}/${tag}-%j.out" \
        --error="${LOG_DIR}/${tag}-%j.err" \
        --export=ALL,CAMPAIGN_TAG="${tag}",CAMPAIGN_BOX_DIR="${bdir}",CAMPAIGN_BOX_A="${box}",CAMPAIGN_ENSEMBLE="${ens}",CAMPAIGN_PS="${ps}",CAMPAIGN_DT_FS="${dt}",CAMPAIGN_X64="${x64}",CAMPAIGN_SEED="${seed}" \
        "${ROOT}/scripts/slurm/dense_dt_campaign/sbatch_one.sh")
      echo "RESUBMITTED $tag -> job $jid  ens=$ens box=${box} dt=${dt} x64=${x64} ps=${ps}" | tee -a "$OUT_ROOT/bench.log" "$OUT_ROOT/job_ids.txt" "$MONITOR_LOG"
    done
  fi

  # If no ddc jobs at all and boxes ready, re-submit full matrix
  if [[ "$n_run" -eq 0 && "$n_pend" -eq 0 ]]; then
    done_n=$(rg -c '^RESULT ' "$OUT_ROOT"/*/bench.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
    if [[ "${done_n:-0}" -lt 8 ]] && box_ready "$BOX24"; then
      actions+=("queue empty with incomplete arms → submit_all.sh")
      bash "${ROOT}/scripts/slurm/dense_dt_campaign/submit_all.sh" | tee -a "$MONITOR_LOG" || true
    fi
  fi
fi

if ((${#actions[@]} > 0)); then
  printf -- '- %s\n' "${actions[@]}" | tee -a "$STATUS" "$MONITOR_LOG"
else
  echo "- (none)" | tee -a "$STATUS" "$MONITOR_LOG"
fi

# Agent reminder breadcrumb
{
  echo
  echo "## Agent TODO (when you wake)"
  echo
  echo "1. Read \`$STATUS\` and \`$MONITOR_LOG\` tail"
  echo "2. If boxes stuck: check packmol tolerance fix + \`/tmp/build_dense_boxes_v3.log\`"
  echo "3. When H5s land: compare E_tot / H_NHC / bond health vs sparse L30"
  echo "4. Manuscript path: density table for DCM hybrid → docs/manuscripts/.../results.md §3"
  echo
} >> "$STATUS"

echo "Wrote $STATUS" | tee -a "$MONITOR_LOG"
exit 0
