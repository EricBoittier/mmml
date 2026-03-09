#!/bin/bash
#SBATCH --job-name=PhysNet_Scan
#SBATCH --time=06:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --array=0-23
#SBATCH --cpus-per-task=2
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --output=logs/scan_%A_%a.out
#SBATCH --error=logs/scan_%A_%a.err

# Load CUDA
module load CUDA/12.2.0

# CHECK GPU AVAILABILITY
echo "=== GPU Check ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================="

source ~/mmml/.venv/bin/activate
# ... rest of script



