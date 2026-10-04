#!/bin/bash
#SBATCH --job-name=spicea-eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-6hours
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/spicea_eval_%j.out
# Score a packed full-SPICE-α checkpoint on whole test/valid splits:
#   sbatch scripts/spice_alpha/eval_full_packed.sbatch CKPT_DIR PARAMS_JSON OUT_DIR [--pack-scale 0.5 ...]
# PARAMS_JSON should be a frozen copy (training overwrites params-best.json every epoch).
set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml_dev}"
PY="${MMML_PYTHON:-$HOME/mmml_new/.venv/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" scripts/spice_alpha/eval_full_packed.py "$@"
