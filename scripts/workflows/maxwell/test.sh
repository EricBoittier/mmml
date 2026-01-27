#!/bin/bash

#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --array=0-120
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090 #a100  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc

hostname
which python
module load cuDNN
module load OpenMM

cdir=$PWD

which python
source ~/mmml/.venv/bin/activate
cd ~/mmml
bash setup/install.sh
cd $cdir
uv run test.py $SLURM_ARRAY_TASK_ID

