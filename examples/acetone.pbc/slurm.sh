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
#module load 
bash 01_make.sh
bash 02_sim.sh

