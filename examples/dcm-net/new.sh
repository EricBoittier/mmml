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

set -eo pipefail

module load Miniconda3
# If your site uses CUDA modules, load the one matching your jaxlib wheel:
# module load CUDA/12.2
module load CUDA/12.2.0
#module load cuDNN/8.9.7.29-CUDA-12.4.0
eval "$(conda shell.bash hook)"

# binutils hook + nounset can clash; keep -u off (or wrap activate with set +u / set -u)
#conda activate mmml-full

# Block user-site to avoid ~/.local/jax_plugins
#export PYTHONNOUSERSITE=1

# JAX memory knobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

source ~/mmml/.venv/bin/activate
uv sync --extra gpu
# Sanity print
uv run python - <<'PY'
import os, jax
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())
try:
    import jaxlib, platform
    from jaxlib import version as jlv
    print("jax =", jax.__version__, "jaxlib =", jaxlib.__version__)
    print("jaxlib CUDA =", getattr(jlv, "__cuda_version__", "unknown"))
except Exception as e:
    print("Version check failed:", e)
PY
uv run which python
# Launch your actual training (fix this line)
uv run python train.py #--config dcm.yaml

