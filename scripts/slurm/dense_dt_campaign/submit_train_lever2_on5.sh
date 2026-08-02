#!/usr/bin/env bash
# Preflight + sbatch for lever-2 on=5 retrain; appends job id to job_ids.txt.
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

echo "Submitting lever-2 on=5 FT"
echo "  config=$CONFIG"
echo "  data=$DATA"
echo "  warmstart=$CKPT"
echo "  epochs=${DDC_ON5_EPOCHS:-50} tag=${DDC_ON5_TAG:-hybrid_mm_lever2_on5_ft}"

JOB_LINE="$(sbatch --export=ALL scripts/slurm/dense_dt_campaign/sbatch_train_lever2_on5.sh)"
echo "$JOB_LINE"
JOB_ID="$(awk '{print $4}' <<<"$JOB_LINE")"
echo "$JOB_ID" | tee -a "$ROOT/artifacts/lj_scales/dense_dt_campaign/job_ids.txt"
echo "logs: $LOG_DIR/ddc-train-on5-${JOB_ID}.{out,err}"
