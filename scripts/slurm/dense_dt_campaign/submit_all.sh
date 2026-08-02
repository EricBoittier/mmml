#!/usr/bin/env bash
# Submit denser-box + dt/x64 ensemble matrix for overnight GPU runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT=artifacts/lj_scales/dense_dt_campaign
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
chmod +x "${ROOT}/scripts/slurm/dense_dt_campaign/"*.sh

echo "dense_dt_campaign submit $(date -Is) host=$(hostname) sha=$(git rev-parse --short HEAD)" | tee -a "$OUT_ROOT/bench.log"

submit() {
  local tag="$1" box_dir="$2" box_a="$3" ens="$4" ps="$5" dt="$6" x64="$7" seed="$8"
  local jid
  jid=$(sbatch --parsable \
    --job-name="ddc-${tag}" \
    --output="${LOG_DIR}/${tag}-%j.out" \
    --error="${LOG_DIR}/${tag}-%j.err" \
    --export=ALL,CAMPAIGN_TAG="${tag}",CAMPAIGN_BOX_DIR="${box_dir}",CAMPAIGN_BOX_A="${box_a}",CAMPAIGN_ENSEMBLE="${ens}",CAMPAIGN_PS="${ps}",CAMPAIGN_DT_FS="${dt}",CAMPAIGN_X64="${x64}",CAMPAIGN_SEED="${seed}" \
    "${ROOT}/scripts/slurm/dense_dt_campaign/sbatch_one.sh")
  echo "SUBMITTED $tag -> job $jid  ens=$ens box=${box_a} dt=${dt} x64=${x64} ps=${ps}" | tee -a "$OUT_ROOT/bench.log" "$OUT_ROOT/job_ids.txt"
}

BOX24=artifacts/lj_scales/liquid_dense_L24
BOX26=artifacts/lj_scales/liquid_dense_L26

# --- Near-bulk L=24 (~1.22 g/cm³ for DCM:120) ---
submit L24_nvt_dt1_f32_50ps   "$BOX24" 24 nvt 50 1.0 0 101
submit L24_nvt_dt05_x64_50ps  "$BOX24" 24 nvt 50 0.5 1 102
submit L24_npt_dt1_f32_50ps   "$BOX24" 24 npt 50 1.0 0 103
submit L24_npt_dt05_x64_50ps  "$BOX24" 24 npt 50 0.5 1 104
submit L24_nve_dt05_x64_20ps  "$BOX24" 24 nve 20 0.5 1 105

# --- Intermediate L=26 (~0.96 g/cm³) ---
submit L26_nvt_dt1_f32_50ps   "$BOX26" 26 nvt 50 1.0 0 201
submit L26_npt_dt1_f32_50ps   "$BOX26" 26 npt 50 1.0 0 202

# --- Control: old sparse L=30 NVT dt0.5 x64 (short) for comparison ---
submit L30_nvt_dt05_x64_20ps  artifacts/lj_scales/liquid_nvt 30 nvt 20 0.5 1 301

echo "All submitted. Track with: squeue -u \$USER -n ddc-" | tee -a "$OUT_ROOT/bench.log"
squeue -u "$USER" -o '%.18i %.12P %.28j %.2t %.10M %R' | head -40
