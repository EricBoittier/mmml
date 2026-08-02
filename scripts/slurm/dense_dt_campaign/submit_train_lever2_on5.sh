#!/usr/bin/env bash
# Preflight + sbatch for lever-2 on=5 retrain; appends job id to job_ids.txt.
#
# Uses a *clean* Slurm export list — never --export=ALL — so a login-shell
# JAX_PLATFORMS=cpu / MMML_JAX_WARMUP_DEVICE=cpu cannot silently CPU-train.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/artifacts/lj_scales/dense_dt_campaign/logs"
mkdir -p "$LOG_DIR"

CONFIG="${DDC_ON5_CONFIG:-examples/hybrid_mm_charges/train_fixed_lj_scales_on5.yaml}"
DATA="${DDC_ON5_DATA:-artifacts/lj_scales/dataset_cgenff.npz}"
CKPT="${DDC_ON5_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json}"

for f in "$CONFIG" "$DATA" "$CKPT"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done

echo "Submitting lever-2 on=5 FT (clean GPU env, exclusive node)"
echo "  config=$CONFIG"
echo "  data=$DATA"
echo "  warmstart=$CKPT"
echo "  epochs=${DDC_ON5_EPOCHS:-50} batch=${DDC_ON5_BATCH:-64} tag=${DDC_ON5_TAG:-hybrid_mm_lever2_on5_ft}"

# Explicit export list only (no ALL): a login-shell JAX_PLATFORMS=cpu must not
# leak into the allocation. Train script also hard-sets CUDA device vars.
EXPORT_LIST="LJ_DEVICE=gpu,JAX_PLATFORMS=cuda,MMML_MLPOT_DEVICE=gpu,MMML_JAX_WARMUP_DEVICE=gpu,MMML_MM_NL_DEVICE=gpu"
EXPORT_LIST+=",DDC_ON5_CONFIG=${CONFIG},DDC_ON5_DATA=${DATA},DDC_ON5_CKPT=${CKPT}"
EXPORT_LIST+=",DDC_ON5_EPOCHS=${DDC_ON5_EPOCHS:-50},DDC_ON5_TAG=${DDC_ON5_TAG:-hybrid_mm_lever2_on5_ft}"
EXPORT_LIST+=",DDC_ON5_N_TRAIN=${DDC_ON5_N_TRAIN:-32000},DDC_ON5_N_VALID=${DDC_ON5_N_VALID:-5950}"
EXPORT_LIST+=",DDC_ON5_SEED=${DDC_ON5_SEED:-42},DDC_ON5_BATCH=${DDC_ON5_BATCH:-64}"
EXPORT_LIST+=",DDC_ON5_CKPT_DIR=${DDC_ON5_CKPT_DIR:-artifacts/lj_scales/ckpts}"
EXPORT_LIST+=",UV_NO_SYNC=1,PYTHONUNBUFFERED=1,PATH=${PATH},HOME=${HOME},USER=${USER:-$LOGNAME}"

JOB_LINE="$(sbatch --export="${EXPORT_LIST}" scripts/slurm/dense_dt_campaign/sbatch_train_lever2_on5.sh)"
echo "$JOB_LINE"
JOB_ID="$(awk '{print $4}' <<<"$JOB_LINE")"
echo "$JOB_ID" | tee -a "$ROOT/artifacts/lj_scales/dense_dt_campaign/job_ids.txt"
echo "logs: $LOG_DIR/ddc-train-on5-${JOB_ID}.{out,err}"
