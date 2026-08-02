#!/usr/bin/env bash
# Submit distilled on=5 FT (clean GPU env — no --export=ALL).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs

CONFIG="${DDC_ON5D_CONFIG:-examples/hybrid_mm_charges/train_fixed_lj_scales_on5_distill.yaml}"
DATA="${DDC_ON5D_DATA:-artifacts/lj_scales/dataset_cgenff.npz}"
CKPT="${DDC_ON5D_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json}"
for f in "$CONFIG" "$DATA" "$CKPT"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done

echo "Submitting distill on=5 FT"
echo "  epochs=${DDC_ON5D_EPOCHS:-15} lr=${DDC_ON5D_LR:-0.0001} alpha=${DDC_ON5D_DISTILL_ALPHA:-0.35}"

EXPORT_LIST="LJ_DEVICE=gpu,JAX_PLATFORMS=cuda,MMML_MLPOT_DEVICE=gpu,MMML_JAX_WARMUP_DEVICE=gpu,MMML_MM_NL_DEVICE=gpu"
EXPORT_LIST+=",DDC_ON5D_CONFIG=${CONFIG},DDC_ON5D_DATA=${DATA},DDC_ON5D_CKPT=${CKPT}"
EXPORT_LIST+=",DDC_ON5D_EPOCHS=${DDC_ON5D_EPOCHS:-15},DDC_ON5D_LR=${DDC_ON5D_LR:-0.0001}"
EXPORT_LIST+=",DDC_ON5D_DISTILL_ALPHA=${DDC_ON5D_DISTILL_ALPHA:-0.35}"
EXPORT_LIST+=",DDC_ON5D_TAG=${DDC_ON5D_TAG:-hybrid_mm_lever2_on5_distill}"
EXPORT_LIST+=",DDC_ON5D_BATCH=${DDC_ON5D_BATCH:-64}"
EXPORT_LIST+=",UV_NO_SYNC=1,PYTHONUNBUFFERED=1,PATH=${PATH},HOME=${HOME},USER=${USER:-$LOGNAME}"

JOB_LINE="$(sbatch --export="${EXPORT_LIST}" scripts/slurm/dense_dt_campaign/sbatch_train_lever2_on5_distill.sh)"
echo "$JOB_LINE"
JOB_ID="$(awk '{print $4}' <<<"$JOB_LINE")"
echo "$JOB_ID" | tee -a artifacts/lj_scales/dense_dt_campaign/job_ids.txt
echo "logs: artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-distill-${JOB_ID}.{out,err}"
