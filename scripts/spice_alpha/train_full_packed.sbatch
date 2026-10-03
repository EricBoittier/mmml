#!/bin/bash
#SBATCH --job-name=spicea-full
#SBATCH --partition=h200
#SBATCH --qos=h200-1day
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/spicea_full_%j.out
# Full SPICE-α packed training on one H200. Options pass through to the driver:
#   sbatch scripts/spice_alpha/train_full_packed.sbatch OUT_DIR [--epochs 20 ...]
set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml_dev}"
PY="${MMML_PYTHON:-$HOME/mmml_new/.venv/bin/python}"
RAGGED="${RAGGED:-$HOME/data/spicealpha/ragged}"
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
OUT="$1"; shift
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader -l 30 > "$OUT/gpu_usage.csv" &
trap 'kill $! 2>/dev/null || true' EXIT
"$PY" scripts/spice_alpha/train_full_packed.py "$RAGGED" "$OUT" "$@"
