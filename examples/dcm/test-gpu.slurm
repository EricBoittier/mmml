#!/bin/bash
 
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --partition=titan
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
 
#module load CUDA
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
hostname
source ~/mmml/.venv/bin/activate
which python
nvidia-smi                       # should show GPUs & driver
python -c "import jax; print(jax.devices())"
python -c "import jax; print(jax.default_backend())"
python -c "import jaxlib; import jax; print(jax.__version__)"
python train.py 


