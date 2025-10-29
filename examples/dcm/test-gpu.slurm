#!/bin/bash
 
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=titan
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
 
module load CUDA
hostname
source ~/mmml/.venv/bin/activate
which python
python train.py 


