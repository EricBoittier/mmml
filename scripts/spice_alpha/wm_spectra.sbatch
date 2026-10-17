#!/bin/bash
#SBATCH --job-name=wm-spectra
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-6hours
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/wm_spectra_%j.out
# IR / Raman / VCD(null) of the water–methanol boxes: sbatch wm_spectra.sbatch OUT --hybrid TAG=RUN ... --efield TAG=TRAJ ... --box TAG=BOX.json ...
set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml_dev}"; PY="${MMML_PYTHON:-$HOME/mmml_new/.venv/bin/python}"
cd "$REPO"; source "$REPO/scripts/spice_alpha/gpu_guard.sh"
export PYTHONPATH="$REPO:$REPO/scripts/spice_alpha" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" scripts/spice_alpha/wm_spectra.py "$@"
