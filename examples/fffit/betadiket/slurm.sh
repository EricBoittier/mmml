#!/bin/bash

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=3000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

hostname

echo $PWD

source ~/mmml/.venv/bin/activate
module load gcc
# Skip CHARMM energy.show() in make_res to avoid segfault in CHARMM bond routines under SLURM
export SKIP_CHARMM_ENERGY_SHOW=1
bash 01_make.sh
bash 02_sim.sh

