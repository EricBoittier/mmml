#!/bin/bash
 
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --partition=titan #a100  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc

module load Miniconda3

conda activate #mmml-gpu

module load CUDA/12.1.1
export CUDA_VISIBLE_DEVICES=0
module purge
conda activate mmml-full
/scicore/soft/easybuild/apps/Miniconda3/24.7.1-0/bin/python train.py

