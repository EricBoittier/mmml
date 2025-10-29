#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=20000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu

hostname

#module load cudnn/cudnn-9.4-gcc-14.1-ompi-5.0-cuda-12.6

nvidia-smi
#nvcc --version


source ~/.bashrc
conda init bash
conda activate jaxe3xcuda11p39

#export XLA_PYTHON_CLIENT_MEM_FRACTION=".95"
echo $XLA_PYTHON_CLIENT_MEM_FRACTION
#echo $CUDA_VISIBLE_DEVICES

which python
echo "GPU ID:" $CUDA_VISIBLE_DEVICES
echo "NDCM $NDCM"

python -c "import jax; print(jax.devices())"

python /pchem-data/meuwly/boittier/home/jaxeq/dcmnet/main.py --type "dipole" --random_seed $RANDOM --n_dcm $NDCM --n_gpu $CUDA_VISIBLE_DEVICES --include_pseudotensors  

