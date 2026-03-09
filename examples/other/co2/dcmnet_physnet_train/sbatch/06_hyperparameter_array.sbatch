#!/bin/bash
#SBATCH --job-name=hparam_sweep
#SBATCH --time=08:00:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu12hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH --output=logs/hparam_sweep_%A_%a.out
#SBATCH --error=logs/hparam_sweep_%A_%a.err

# Hyperparameter sweep using SLURM job arrays
# Purpose: Test multiple configurations in parallel

set -e  # Exit on any error

echo "=========================================="
echo "Hyperparameter Sweep - Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust as needed for Scicore)
module purge
module load Python/3.11

# Activate conda environment
source activate mmml

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
TRAIN_EFD="train.npz"
TRAIN_ESP="grids_train.npz"
VALID_EFD="valid.npz"
VALID_ESP="grids_valid.npz"

# ============================================
# Hyperparameter Configurations
# ============================================
# Define arrays for different hyperparameters
LEARNING_RATES=(0.001 0.0005 0.0001)
BATCH_SIZES=(4 8 16)
OPTIMIZERS=(adam adamw muon)

# Calculate configuration from array task ID
LR_IDX=$(($SLURM_ARRAY_TASK_ID / 3))
BS_IDX=$(($SLURM_ARRAY_TASK_ID % 3))
OPT="adamw"  # Fixed optimizer

LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

NAME="hparam_lr${LR}_bs${BS}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo ""
echo "Configuration for this array task:"
echo "  Learning rate: $LR"
echo "  Batch size: $BS"
echo "  Optimizer: $OPT"
echo "  Model name: $NAME"
echo ""

# ============================================
# Run Training
# ============================================
python trainer.py \
    --train-efd "$TRAIN_EFD" \
    --train-esp "$TRAIN_ESP" \
    --valid-efd "$VALID_EFD" \
    --valid-esp "$VALID_ESP" \
    --epochs 100 \
    --batch-size "$BS" \
    --learning-rate "$LR" \
    --name "$NAME" \
    --optimizer "$OPT" \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Configuration: LR=$LR, BS=$BS, OPT=$OPT"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

