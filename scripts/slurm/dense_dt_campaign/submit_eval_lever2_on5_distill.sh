#!/usr/bin/env bash
# Submit epoch contact-ok sweep for the newest distill FT run.
# Optional: DDC_ON5D_RUN_DIR=... DDC_ON5D_SWEEP_EPOCHS=1,5,10,15
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs

EXPORT_LIST="LJ_DEVICE=gpu,JAX_PLATFORMS=cuda,MMML_MLPOT_DEVICE=gpu,MMML_JAX_WARMUP_DEVICE=gpu"
EXPORT_LIST+=",DDC_ON5D_TAG=${DDC_ON5D_TAG:-hybrid_mm_lever2_on5_distill}"
EXPORT_LIST+=",DDC_ON5D_SWEEP_EPOCHS=${DDC_ON5D_SWEEP_EPOCHS:-1,3,5,8,10,12,15}"
EXPORT_LIST+=",UV_NO_SYNC=1,PYTHONUNBUFFERED=1,PATH=${PATH},HOME=${HOME},USER=${USER:-$LOGNAME}"
if [[ -n "${DDC_ON5D_RUN_DIR:-}" ]]; then
  EXPORT_LIST+=",DDC_ON5D_RUN_DIR=${DDC_ON5D_RUN_DIR}"
fi

JOB_LINE="$(sbatch --export="${EXPORT_LIST}" scripts/slurm/dense_dt_campaign/sbatch_eval_lever2_on5_distill.sh)"
echo "$JOB_LINE"
JOB_ID="$(awk '{print $4}' <<<"$JOB_LINE")"
echo "$JOB_ID" | tee -a artifacts/lj_scales/dense_dt_campaign/job_ids.txt
echo "logs: artifacts/lj_scales/dense_dt_campaign/logs/ddc-eval-distill-${JOB_ID}.{out,err}"
