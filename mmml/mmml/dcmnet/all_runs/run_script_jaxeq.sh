#!/bin/bash

#SBATCH --mail-user=ericdavid.boittier@unibas.ch
#SBATCH --job-name=jaxeq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=3000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu20

hostname

module load gcc/gcc4.8.5-openmpi1.10-cuda9.2

conda activate /home/boittier/psi4conda/envs/jax_ex
conda init 

python esp_net.py --n_dcm 1 --n_gpu '0' > test.out


