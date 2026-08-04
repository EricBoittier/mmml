#!/usr/bin/env bash
# Dense NVE in PBC with the softwell on=5 deploy ckpt (not epoch222).
#
# Same recipe as the ep222 melt probes (dt=0.25 fs, x64, 5 ps, continue from
# dense NVT frame 0). Pass = no melt (T≲600 K, bonds intact) + ΔE gate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT=artifacts/lj_scales/dense_dt_campaign
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
chmod +x "${ROOT}/scripts/slurm/dense_dt_campaign/"*.sh

export DDC_CKPT="${DDC_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_lever2_on5_softwell_2026-08-02_22-15-54.json}"
export DDC_SIDECAR="${DDC_SIDECAR:-artifacts/lj_scales/ckpts/hybrid_mm_lever2_on5_softwell-657cb7db-74a1-4623-84a5-f772b8fe7928/hybrid_mm.json}"
export DDC_HANDOFF="${DDC_HANDOFF:-soft}"
export DDC_MM_SWITCH_ON="${DDC_MM_SWITCH_ON:-5.0}"
export DDC_ML_SWITCH_WIDTH="${DDC_ML_SWITCH_WIDTH:-1.5}"
export DDC_MM_SWITCH_WIDTH="${DDC_MM_SWITCH_WIDTH:-5.0}"

echo "softwell dense NVE submit $(date -Is) host=$(hostname) sha=$(git rev-parse --short HEAD)" | tee -a "$OUT_ROOT/bench.log"
echo "DDC_CKPT=$DDC_CKPT" | tee -a "$OUT_ROOT/bench.log"
echo "DDC_SIDECAR=$DDC_SIDECAR" | tee -a "$OUT_ROOT/bench.log"

submit() {
  local tag="$1" box_dir="$2" box_a="$3" ens="$4" ps="$5" dt="$6" x64="$7" seed="$8"
  local jid
  jid=$(sbatch --parsable \
    --job-name="ddc-${tag}" \
    --time=08:00:00 \
    --output="${LOG_DIR}/${tag}-%j.out" \
    --error="${LOG_DIR}/${tag}-%j.err" \
    --export=ALL,CAMPAIGN_TAG="${tag}",CAMPAIGN_BOX_DIR="${box_dir}",CAMPAIGN_BOX_A="${box_a}",CAMPAIGN_ENSEMBLE="${ens}",CAMPAIGN_PS="${ps}",CAMPAIGN_DT_FS="${dt}",CAMPAIGN_X64="${x64}",CAMPAIGN_SEED="${seed}",DDC_CKPT="${DDC_CKPT}",DDC_SIDECAR="${DDC_SIDECAR}",DDC_HANDOFF="${DDC_HANDOFF}",DDC_MM_SWITCH_ON="${DDC_MM_SWITCH_ON}",DDC_ML_SWITCH_WIDTH="${DDC_ML_SWITCH_WIDTH}",DDC_MM_SWITCH_WIDTH="${DDC_MM_SWITCH_WIDTH}" \
    "${ROOT}/scripts/slurm/dense_dt_campaign/sbatch_one.sh")
  echo "SUBMITTED $tag -> job $jid  ens=$ens box=${box_a} dt=${dt} x64=${x64} ps=${ps} ckpt=softwell" | tee -a "$OUT_ROOT/bench.log" "$OUT_ROOT/job_ids.txt"
}

BOX24=artifacts/lj_scales/liquid_dense_L24
BOX26=artifacts/lj_scales/liquid_dense_L26

# Match the ep222 melt-probe recipe so results are comparable.
submit L24_nve_dt025_x64_5ps_softwell "$BOX24" 24 nve 5 0.25 1 401
submit L26_nve_dt025_x64_5ps_softwell "$BOX26" 26 nve 5 0.25 1 402

echo "Track: squeue -u \$USER -n ddc-L24_nve_dt025_x64_5ps_softwell,ddc-L26_nve_dt025_x64_5ps_softwell"
squeue -u "$USER" -o '%.18i %.12P %.40j %.2t %.10M %R' | head -20
