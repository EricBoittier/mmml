#!/usr/bin/env bash
#SBATCH --job-name=qcml-mbd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qcml-mbd-%j.out

set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
PY="${MMML_PYTHON:-$ROOT/.venv/bin/python}"
CACHE="${CACHE:-$HOME/orbax_cache/qcml_mbd}"
WORKDIR="${WORKDIR:-$HOME/qcml_runs/mbd_pilot}"
MAX_STRUCTURES="${MAX_STRUCTURES:-500000}"
MAX_ATOMS="${MAX_ATOMS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUCKET_WIDTH="${BUCKET_WIDTH:-4}"
EPOCHS="${EPOCHS:-20}"
VALIDATION_SHARDS="${VALIDATION_SHARDS:-2}"
TEST_SHARDS="${TEST_SHARDS:-2}"
EXCLUDE_NEWEST="${EXCLUDE_NEWEST:-1}"

cd "$ROOT"
mkdir -p logs "$WORKDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

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

"$PY" scripts/train_qcml_mbd.py \
  --cache "$CACHE" \
  --workdir "$WORKDIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-structures "$MAX_STRUCTURES" \
  --max-atoms "$MAX_ATOMS" \
  --bucket-width "$BUCKET_WIDTH" \
  --validation-shards "$VALIDATION_SHARDS" \
  --test-shards "$TEST_SHARDS" \
  --save-every 5 \
  --features 64 \
  --num-iterations 3 \
  --num-basis-functions 16 \
  --cutoff 12.0
