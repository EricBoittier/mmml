#!/bin/bash
 
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=a100  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
 
module load CUDA

hostname

for i in {0..10000}; do echo $i; done




