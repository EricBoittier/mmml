#!/bin/bash
#SBATCH --job-name=dcm-train
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# 1) Modules (if applicable)
# module purge
# module load CUDA/12.2

# 2) Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmml

# 3) (Optional) JAX memory behavior
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

# 4) Quick GPU sanity check
python - <<'PY'
import jax
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())
PY

# 5) Launch your training (no hardcoded CUDA_VISIBLE_DEVICES)
python -m your_package.train --config your_config.yaml

