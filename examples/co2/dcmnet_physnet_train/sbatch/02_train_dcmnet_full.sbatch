#!/bin/bash
#SBATCH --job-name=train_dcmnet_full
#SBATCH --time=24:00:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu24hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_dcmnet_full_%j.out
#SBATCH --error=logs/train_dcmnet_full_%j.err

# Full training run for DCMNet model (200+ epochs)
# Purpose: Production-quality model training

set -e  # Exit on any error

echo "=========================================="
echo "DCMNet Training - Full Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust as needed for Scicore)
module purge
module load Python/3.11

# Activate conda environment
source activate mmml

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"
echo "GPU devices:"
python -c "import jax; print('GPUs:', jax.devices())"
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
TRAIN_EFD="train.npz"
TRAIN_ESP="grids_train.npz"
VALID_EFD="valid.npz"
VALID_ESP="grids_valid.npz"

# Training parameters
EPOCHS=200
BATCH_SIZE=8
NAME="dcmnet_full_${SLURM_JOB_ID}"
OPTIMIZER="adamw"  # Options: adam, adamw, rmsprop, muon

# ============================================
# Run Training
# ============================================
echo "Starting full training with:"
echo "  Train EFD: $TRAIN_EFD"
echo "  Train ESP: $TRAIN_ESP"
echo "  Valid EFD: $VALID_EFD"
echo "  Valid ESP: $VALID_ESP"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Optimizer: $OPTIMIZER"
echo "  Model name: $NAME"
echo ""

python trainer.py \
    --train-efd "$TRAIN_EFD" \
    --train-esp "$TRAIN_ESP" \
    --valid-efd "$VALID_EFD" \
    --valid-esp "$VALID_ESP" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --name "$NAME" \
    --optimizer "$OPTIMIZER" \
    --use-recommended-hparams \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Duration: $(($(date +%s) - SECONDS)) seconds"
echo "=========================================="

# Print checkpoint location
echo ""
echo "Checkpoint saved to: checkpoints/$NAME/"
echo ""

exit $EXIT_CODE

