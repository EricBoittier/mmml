#!/bin/bash
#SBATCH --job-name=GPU_Test
#SBATCH --time=1-00:00:00
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu1day
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --array=0-10

echo "=========================================="
echo "GPU Test Script"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

module activate Miniconda3
conda init bash
conda activate mmml-gpu

export NDCM=1

export seed=$SLURM_JOB_ID
echo $seed 
RunPython="/scicore/home/meuwly/boitti0000/.conda/envs/mmml-gpu/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" && $RunPython compare_models.py \
    --train-efd energies_forces_dipoles_train.npz \
    --train-esp grids_esp_train.npz \
    --valid-efd energies_forces_dipoles_valid.npz \
    --valid-esp grids_esp_valid.npz \
    --epochs 10 \
    --n-dcm $NDCM \
    --batch-size 100 \
    --comparison-name dcm1test$seed \
    --seed $seed

