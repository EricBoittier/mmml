#!/bin/bash
 
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=titan #rtx4090
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
 
module load CUDA #-12.1.1
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
hostname
source ~/mmml/.venv/bin/activate
which python
#unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
nvidia-smi                       # should show GPUs & driver
#uv run python -c "import jax; print(jax.devices())"
#uv run python -c "import jax; print(jax.default_backend())"
#uv run python -c "import jaxlib; import jax; print(jax.__version__)"
uv run python train.py 


