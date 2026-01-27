#!/bin/bash
#SBATCH --job-name=dcm-gpu
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=slurm-%j.out

set -euo pipefail

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate mmml-full    # your env; must already exist

# Optional CUDA module if your site requires it (match your jaxlib build)
# module load CUDA/12.2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

python - <<'PY'
import os, jax
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())
PY

python -m your_package.train --config your_config.yaml

