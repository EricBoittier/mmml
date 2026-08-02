#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs

EXPORT_LIST="LJ_DEVICE=gpu,JAX_PLATFORMS=cuda,MMML_MLPOT_DEVICE=gpu"
EXPORT_LIST+=",DDC_ON5D_TAG=${DDC_ON5SW_TAG:-hybrid_mm_lever2_on5_softwell}"
EXPORT_LIST+=",UV_NO_SYNC=1,PYTHONUNBUFFERED=1,PATH=${PATH},HOME=${HOME},USER=${USER:-$LOGNAME}"
if [[ -n "${DDC_ON5SW_RUN_DIR:-}" ]]; then
  EXPORT_LIST+=",DDC_ON5D_RUN_DIR=${DDC_ON5SW_RUN_DIR}"
fi

JOB_LINE="$(sbatch --export="${EXPORT_LIST}" scripts/slurm/dense_dt_campaign/sbatch_eval_lever2_on5_softwell.sh)"
echo "$JOB_LINE"
JOB_ID="$(awk '{print $4}' <<<"$JOB_LINE")"
echo "$JOB_ID" | tee -a artifacts/lj_scales/dense_dt_campaign/job_ids.txt
echo "logs: artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-sw-eval-${JOB_ID}.{out,err}"
