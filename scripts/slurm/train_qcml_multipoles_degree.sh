#!/usr/bin/env bash
#SBATCH --job-name=qcml-mul-deg
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:30:00
#SBATCH --output=logs/qcml-multipoles-l%j.out

set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
PY="${MMML_PYTHON:-$ROOT/.venv/bin/python}"
CACHE="${CACHE:-$HOME/orbax_cache/qcml_multipoles_traceless}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
TARGET_DEGREE="${TARGET_DEGREE:?Set TARGET_DEGREE to 1, 2, or 3}"
WORKDIR="${WORKDIR:-$HOME/qcml_runs/multipoles_l${TARGET_DEGREE}_${RUN_TAG}}"
MAX_STRUCTURES="${MAX_STRUCTURES:-}"
MAX_ATOMS="${MAX_ATOMS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUCKET_WIDTH="${BUCKET_WIDTH:-4}"
EPOCHS="${EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAVE_OPT_STATE="${SAVE_OPT_STATE:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
TARGET_SCALE_MODE="${TARGET_SCALE_MODE:-q95}"
OUTLIER_QUANTILE="${OUTLIER_QUANTILE:-0.95}"
OUTLIER_DEGREE_MODE="${OUTLIER_DEGREE_MODE:-component}"
COMPOSE_DIPOLE_FROM_ATOMIC="${COMPOSE_DIPOLE_FROM_ATOMIC:-1}"
ENFORCE_TOTAL_CHARGE="${ENFORCE_TOTAL_CHARGE:-1}"
VALIDATION_SHARDS="${VALIDATION_SHARDS:-2}"
TEST_SHARDS="${TEST_SHARDS:-2}"
EXCLUDE_NEWEST="${EXCLUDE_NEWEST:-1}"
FEATURES="${FEATURES:-64}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
NUM_BASIS_FUNCTIONS="${NUM_BASIS_FUNCTIONS:-16}"
CUTOFF="${CUTOFF:-6.0}"

case "$TARGET_DEGREE" in
  1) DEGREE_WEIGHTS="${DEGREE_WEIGHTS:-0:1:0:0}" ;;
  2) DEGREE_WEIGHTS="${DEGREE_WEIGHTS:-0:0:1:0}" ;;
  3) DEGREE_WEIGHTS="${DEGREE_WEIGHTS:-0:0:0:1}" ;;
  *) echo "TARGET_DEGREE must be 1, 2, or 3; got $TARGET_DEGREE" >&2; exit 2 ;;
esac

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
echo "TARGET_DEGREE=$TARGET_DEGREE DEGREE_WEIGHTS=$DEGREE_WEIGHTS"
echo "MAX_STRUCTURES=${MAX_STRUCTURES:-all}"
echo "MAX_ATOMS=$MAX_ATOMS BATCH_SIZE=$BATCH_SIZE BUCKET_WIDTH=$BUCKET_WIDTH"
echo "EPOCHS=$EPOCHS SAVE_EVERY=$SAVE_EVERY SAVE_OPT_STATE=$SAVE_OPT_STATE"
echo "LEARNING_RATE=$LEARNING_RATE WEIGHT_DECAY=$WEIGHT_DECAY"
echo "GRADIENT_CLIP_NORM=$GRADIENT_CLIP_NORM HUBER_DELTA=$HUBER_DELTA"
echo "TARGET_SCALE_MODE=$TARGET_SCALE_MODE OUTLIER_QUANTILE=$OUTLIER_QUANTILE"
echo "OUTLIER_DEGREE_MODE=$OUTLIER_DEGREE_MODE"
echo "COMPOSE_DIPOLE_FROM_ATOMIC=$COMPOSE_DIPOLE_FROM_ATOMIC ENFORCE_TOTAL_CHARGE=$ENFORCE_TOTAL_CHARGE"

"$PY" scripts/snapshot_orbax_shard_manifest.py \
  --cache "$CACHE" \
  --dataset-kind qcml_multipoles \
  --exclude-newest "$EXCLUDE_NEWEST" \
  --output "$CACHE/manifest.json"

"$PY" scripts/audit_qcml_shards.py \
  --cache "$CACHE" \
  --kind multipoles \
  --max-shards 2 \
  --samples-per-shard 1000 \
  --output "$WORKDIR/shard_audit.json"

train_args=(
  --cache "$CACHE"
  --workdir "$WORKDIR"
  --target-degree "$TARGET_DEGREE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --gradient-clip-norm "$GRADIENT_CLIP_NORM"
  --charge-weight 0.0
  --huber-delta "$HUBER_DELTA"
  --degree-weights "$DEGREE_WEIGHTS"
  --target-scale-mode "$TARGET_SCALE_MODE"
  --outlier-quantile "$OUTLIER_QUANTILE"
  --outlier-degree-mode "$OUTLIER_DEGREE_MODE"
  --max-atoms "$MAX_ATOMS"
  --bucket-width "$BUCKET_WIDTH"
  --validation-shards "$VALIDATION_SHARDS"
  --test-shards "$TEST_SHARDS"
  --save-every "$SAVE_EVERY"
  --features "$FEATURES"
  --num-iterations "$NUM_ITERATIONS"
  --num-basis-functions "$NUM_BASIS_FUNCTIONS"
  --cutoff "$CUTOFF"
)

if [[ -n "$MAX_STRUCTURES" ]]; then
  train_args+=(--max-structures "$MAX_STRUCTURES")
fi
if [[ "$SAVE_OPT_STATE" == "1" ]]; then
  train_args+=(--save-opt-state)
fi
if [[ "$COMPOSE_DIPOLE_FROM_ATOMIC" == "1" ]]; then
  train_args+=(--compose-dipole-from-atomic)
fi
if [[ "$ENFORCE_TOTAL_CHARGE" != "1" ]]; then
  train_args+=(--no-enforce-total-charge)
fi

"$PY" scripts/train_qcml_multipoles.py "${train_args[@]}"
