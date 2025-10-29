#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu

hostname

source ~/.bashrc

conda init bash

conda activate jaxe3xcuda11p39

which python
echo "GPU ID:" $CUDA_VISIBLE_DEVICES
python /pchem-data/meuwly/boittier/home/jaxeq/dcmnet/main.py --type "dipole" --random_seed $RANDOM --n_dcm 3 --n_gpu $CUDA_VISIBLE_DEVICES --include_pseudotensors  

