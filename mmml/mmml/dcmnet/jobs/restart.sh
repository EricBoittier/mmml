#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=5000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu02

hostname

source ~/.bashrc

conda init bash

conda activate jaxe3xcuda11p39

which python
echo "GPU ID:" $CUDA_VISIBLE_DEVICES

echo "restart" $restart

#restart="/pchem-data/meuwly/boittier/home/jaxeq/all_runs/runs11/"
python /pchem-data/meuwly/boittier/home/jaxeq/dcmnet/main.py --type "dipole" --random_seed $RANDOM --n_gpu $CUDA_VISIBLE_DEVICES --restart $restart

