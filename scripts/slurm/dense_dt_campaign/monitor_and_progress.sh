#!/usr/bin/env bash
# Watchdog for denser-box + dt/x64 campaign (manuscript §§7–8: conservation + liquid ρ).
# Usage:
#   bash scripts/slurm/dense_dt_campaign/monitor_and_progress.sh           # report
#   bash scripts/slurm/dense_dt_campaign/monitor_and_progress.sh --react   # report + remediate
set -euo pipefail

# Cron often has a minimal env — force identity + a usable PATH.
export USER="${USER:-$(id -un 2>/dev/null || echo boittier)}"
export HOME="${HOME:-/mmhome/boittier/home}"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:/usr/bin:/bin:${PATH:-/usr/bin:/bin}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT=artifacts/lj_scales/dense_dt_campaign
LOG_DIR="${OUT_ROOT}/logs"
STATUS="${OUT_ROOT}/STATUS.md"
MONITOR_LOG="${OUT_ROOT}/monitor.log"
MARKER_BOX_BUILD=/tmp/build_dense_boxes_v3.sh
REACT=0
[[ "${1:-}" == "--react" ]] && REACT=1

# Prefer ripgrep when present; fall back to grep -E (cron PATH may lack rg).
if command -v rg >/dev/null 2>&1; then
  _match() { rg -N "$@"; }
  _count_result() { rg -c '^RESULT ' "$@" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}'; }
else
  _match() { grep -E "$@"; }
  _count_result() { grep -ch '^RESULT ' "$@" 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"
ts="$(date -Is)"
host="$(hostname)"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

{
  echo "## dense_dt_campaign monitor $ts host=$host sha=$sha react=$REACT user=$USER"
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

# Latest job id per tag from job_ids.txt (avoid resubmitting from stale CANCELLED rows).
declare -A LATEST_JID=()
declare -A LATEST_META=()
if [[ -f "$OUT_ROOT/job_ids.txt" ]]; then
  while read -r _line || [[ -n "${_line:-}" ]]; do
    [[ "$_line" == SUBMITTED* || "$_line" == RESUBMITTED* ]] || continue
    tag=$(awk '{print $2}' <<<"$_line")
    jid=$(awk '{for(i=1;i<=NF;i++) if($i=="job"){print $(i+1); exit}}' <<<"$_line")
    [[ -n "${tag:-}" && "$jid" =~ ^[0-9]+$ ]] || continue
    LATEST_JID["$tag"]="$jid"
    LATEST_META["$tag"]="$_line"
  done < "$OUT_ROOT/job_ids.txt"
fi

declare -a NEED_RESUBMIT=()
for tag in "${!LATEST_JID[@]}"; do
  jid="${LATEST_JID[$tag]}"
  state=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)
  if [[ "$state" == COMPLETED ]]; then
    mkdir -p "$OUT_ROOT/${tag}"
    # Only SUCCESS when the arm's RESULT is rc=0, the per-tag log exists, and
    # it is not a science blow-up (Slurm COMPLETED alone is not enough).
    arm_log="$OUT_ROOT/${tag}/bench.log"
    res_line=""
    [[ -f "$OUT_ROOT/bench.log" ]] && res_line="$(_match "^RESULT ${tag} " "$OUT_ROOT/bench.log" 2>/dev/null | tail -1 || true)"
    [[ -z "$res_line" && -f "$arm_log" ]] \
      && res_line="$(_match "^RESULT " "$arm_log" 2>/dev/null | tail -1 || true)"
    blew=0
    if [[ ! -s "$arm_log" ]]; then
      # Missing/empty arm log: cannot verify; never auto-SUCCESS from rc alone.
      blew=1
    else
      _match "energy blow-up|Partial output saved after error" "$arm_log" >/dev/null 2>&1 && blew=1
    fi
    if [[ "$res_line" == *" rc=0 "* && "$blew" -eq 0 ]]; then
      touch "$OUT_ROOT/${tag}/SUCCESS.flag" 2>/dev/null || true
    else
      rm -f "$OUT_ROOT/${tag}/SUCCESS.flag" 2>/dev/null || true
    fi
    continue
  fi
  if [[ "$state" == FAILED || "$state" == TIMEOUT || "$state" == NODE_FAIL ]]; then
    if [[ ! -f "$OUT_ROOT/${tag}/SUCCESS.flag" ]] \
      && ! squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "ddc-${tag}"; then
      NEED_RESUBMIT+=("$tag")
    fi
  fi
done

# --- Box build process ---
# Only count packmol / box-build that belong to this campaign. A long-running
# packmol under lj_scales_des_validation/.../meoh must not trigger dense rebuilds
# or get killed by --react.
box_build_alive=0
pgrep -f 'build_dense_boxes_v[23]\.sh|liquid-box .*liquid_dense_L' >/dev/null 2>&1 && box_build_alive=1
packmol_alive=0
packmol_stuck=0
while IFS= read -r _pk; do
  [[ -z "${_pk:-}" ]] && continue
  _cwd="$(readlink -f "/proc/${_pk}/cwd" 2>/dev/null || true)"
  case "${_cwd}" in
    *liquid_dense_L24*|*liquid_dense_L26*|*dense_dt_campaign*)
      packmol_alive=1
      _etime=$(ps -p "${_pk}" -o etimes= 2>/dev/null | awk '{print $1+0}')
      if [[ -n "${_etime:-}" && "${_etime}" -gt 2700 ]] && ! box_ready "$BOX24"; then
        packmol_stuck=1
      fi
      ;;
  esac
done < <(pgrep -x packmol 2>/dev/null || true)

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
    [[ "$tag" == logs || "$tag" == plots ]] && continue
    res=$(_match '^RESULT ' "$dir/bench.log" 2>/dev/null | tail -1 || true)
    nh5=$(find "$dir" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')
    fail=""
    [[ -f "$dir/FAIL.txt" ]] && fail="FAIL.txt"
    [[ -f "$dir/SUCCESS.flag" ]] && fail="${fail:+$fail }SUCCESS"
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
  if [[ -f "$OUT_ROOT/PAUSE_RESUBMIT" ]]; then
    actions+=("PAUSE_RESUBMIT set — skipping box rebuild and job resubmits")
    REACT=0
  fi
fi

if [[ "$REACT" -eq 1 ]]; then
  if [[ "$packmol_stuck" -eq 1 ]]; then
    actions+=("kill stuck dense-campaign packmol (>45min) and restart dense box build v3")
    while IFS= read -r _pid; do
      [[ -n "${_pid:-}" ]] && kill "${_pid}" 2>/dev/null || true
    done < <(pgrep -f 'build_dense_boxes_v[23]\.sh' 2>/dev/null || true)
    # Never pkill -x packmol globally — other campaigns (e.g. MeOH DES) may be packing.
    while IFS= read -r _pk; do
      [[ -z "${_pk:-}" ]] && continue
      _cwd="$(readlink -f "/proc/${_pk}/cwd" 2>/dev/null || true)"
      case "${_cwd}" in
        *liquid_dense_L24*|*liquid_dense_L26*|*dense_dt_campaign*)
          kill "${_pk}" 2>/dev/null || true
          ;;
      esac
    done < <(pgrep -x packmol 2>/dev/null || true)
    sleep 2
    box_build_alive=0
  fi

  if [[ "$box_build_alive" -eq 0 ]] && { ! box_ready "$BOX24" || ! box_ready "$BOX26"; }; then
    actions+=("start/restart dense box build (L24 then L26, packmol-tol=1.5)")
    export MMML_DENSE_DT_ROOT="$ROOT"
    cat > "$MARKER_BOX_BUILD" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
cd "${MMML_DENSE_DT_ROOT:-/mmhome/boittier/home/mmml}"
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
  return $rc
}
build 24 1.15 artifacts/lj_scales/liquid_dense_L24
build 26 0.96 artifacts/lj_scales/liquid_dense_L26
echo "ALL BOXES DONE $(date -Is)" | tee -a "$LOG"
EOS
    chmod +x "$MARKER_BOX_BUILD"
    SESSION=dense-box-build
    tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$SESSION" 2>/dev/null && tmux -f /exec-daemon/tmux.portal.conf kill-session -t "$SESSION" 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$SESSION" -c "$ROOT" -- bash "$MARKER_BOX_BUILD"
  fi

  if ((${#NEED_RESUBMIT[@]} > 0)); then
    for tag in "${NEED_RESUBMIT[@]}"; do
      line="${LATEST_META[$tag]:-}"
      [[ -z "$line" ]] && continue
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
      # Archive prior failed outputs so the new run starts clean.
      if [[ -d "$OUT_ROOT/${tag}" && ! -f "$OUT_ROOT/${tag}/SUCCESS.flag" ]]; then
        stamp=$(date +%Y%m%dT%H%M%S)
        mv "$OUT_ROOT/${tag}" "$OUT_ROOT/${tag}.fail_${stamp}" 2>/dev/null || true
      fi
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
fi

if ((${#actions[@]} > 0)); then
  printf -- '- %s\n' "${actions[@]}" | tee -a "$STATUS" "$MONITOR_LOG"
else
  echo "- (none)" | tee -a "$STATUS" "$MONITOR_LOG"
fi

{
  echo
  echo "## Science compares (do not drop)"
  echo
  if [[ -f "$OUT_ROOT/NVE_COMPARE.md" ]]; then
    echo "### NVE (§7) — see also \`NVE_COMPARE.md\` / \`AGENT_STATUS.md\`"
    echo
    # table + takeaway bullets (avoid printing Takeaway header twice)
    awk '/^\| tag /{p=1} p; /^## Takeaway$/{getline; while(NF){print; if(!getline) exit} exit}' \
      "$OUT_ROOT/NVE_COMPARE.md" | head -40
    echo
  fi
  if [[ -f "$OUT_ROOT/NVT_COMPARE.md" ]]; then
    echo "### NVT (§8 proxy) — see \`NVT_COMPARE.md\`"
    echo
    awk '/^\| tag /{p=1} p; /^## Takeaway$/{getline; while(NF){print; if(!getline) exit} exit}' \
      "$OUT_ROOT/NVT_COMPARE.md" | head -40
    echo
  fi
  echo "## Agent TODO (when you wake)"
  echo
  echo "1. Read \`$STATUS\`, \`AGENT_STATUS.md\`, and \`$MONITOR_LOG\` tail"
  echo "2. Plots for passed NVT: \`$OUT_ROOT/plots/\`"
  echo "3. NVE H5 compares written: \`NVE_COMPARE.md\` (dense melt vs L30 conserve)"
  echo "4. NPT still blocked by ~1 ps blow-up — keep \`PAUSE_RESUBMIT\` until SHAKE/softer barostat"
  echo
} >> "$STATUS"

echo "Wrote $STATUS" | tee -a "$MONITOR_LOG"
exit 0
