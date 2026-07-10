#!/usr/bin/env bash
#SBATCH --job-name=qcml-mbd-r
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:30:00
#SBATCH --output=logs/qcml-mbd-restart-%j.out

set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
PY="${MMML_PYTHON:-$ROOT/.venv/bin/python}"
CACHE="${CACHE:-$HOME/orbax_cache/qcml_mbd}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
WORKDIR="${MBD_WORKDIR:-${WORKDIR:-$HOME/qcml_runs/mbd_restart_${RUN_TAG}}}"
MAX_STRUCTURES="${MAX_STRUCTURES:-}"
MAX_ATOMS="${MAX_ATOMS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUCKET_WIDTH="${BUCKET_WIDTH:-4}"
EPOCHS="${EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SAVE_OPT_STATE="${SAVE_OPT_STATE:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
VALIDATION_SHARDS="${VALIDATION_SHARDS:-2}"
TEST_SHARDS="${TEST_SHARDS:-2}"
EXCLUDE_NEWEST="${EXCLUDE_NEWEST:-1}"

cd "$ROOT"
mkdir -p logs "$WORKDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
if [[ "${DISABLE_PINNED_HOST_TRANSFER:-0}" == "1" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_pinned_host_transfer=false"
fi

echo "ROOT=$ROOT"
echo "CACHE=$CACHE"
echo "WORKDIR=$WORKDIR"
echo "MAX_STRUCTURES=${MAX_STRUCTURES:-all}"
echo "MAX_ATOMS=$MAX_ATOMS BATCH_SIZE=$BATCH_SIZE BUCKET_WIDTH=$BUCKET_WIDTH"
echo "EPOCHS=$EPOCHS SAVE_EVERY=$SAVE_EVERY"
echo "SAVE_OPT_STATE=$SAVE_OPT_STATE"
echo "LEARNING_RATE=$LEARNING_RATE WEIGHT_DECAY=$WEIGHT_DECAY"

"$PY" scripts/snapshot_orbax_shard_manifest.py \
  --cache "$CACHE" \
  --dataset-kind qcml_mbd \
  --exclude-newest "$EXCLUDE_NEWEST" \
  --output "$CACHE/manifest.json"

"$PY" scripts/audit_qcml_shards.py \
  --cache "$CACHE" \
  --kind mbd \
  --max-shards 2 \
  --samples-per-shard 1000 \
  --output "$WORKDIR/shard_audit.json"

train_args=(
  --cache "$CACHE"
  --workdir "$WORKDIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --max-atoms "$MAX_ATOMS"
  --bucket-width "$BUCKET_WIDTH"
  --validation-shards "$VALIDATION_SHARDS"
  --test-shards "$TEST_SHARDS"
  --save-every "$SAVE_EVERY"
  --features 64
  --num-iterations 3
  --num-basis-functions 16
  --cutoff 12.0
)

if [[ -n "$MAX_STRUCTURES" ]]; then
  train_args+=(--max-structures "$MAX_STRUCTURES")
fi
if [[ "$SAVE_OPT_STATE" == "1" ]]; then
  train_args+=(--save-opt-state)
fi

"$PY" scripts/train_qcml_mbd.py "${train_args[@]}"
