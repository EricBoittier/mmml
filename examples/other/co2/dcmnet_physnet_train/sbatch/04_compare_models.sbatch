#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --time=12:00:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu12hours
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/compare_models_%j.out
#SBATCH --error=logs/compare_models_%j.err

# Model Comparison - DCMNet vs Non-Equivariant
# Purpose: Head-to-head comparison with equivariance testing

set -e  # Exit on any error

echo "=========================================="
echo "Model Comparison - DCMNet vs Non-Equivariant"
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

# Comparison parameters
EPOCHS=100
BATCH_SIZE=4
COMPARISON_NAME="model_comparison_${SLURM_JOB_ID}"
EQ_SAMPLES=50  # Number of equivariance test samples

# ============================================
# Run Comparison
# ============================================
echo "Starting model comparison with:"
echo "  Train EFD: $TRAIN_EFD"
echo "  Train ESP: $TRAIN_ESP"
echo "  Valid EFD: $VALID_EFD"
echo "  Valid ESP: $VALID_ESP"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Comparison name: $COMPARISON_NAME"
echo "  Equivariance samples: $EQ_SAMPLES"
echo ""

python compare_models.py \
    --train-efd "$TRAIN_EFD" \
    --train-esp "$TRAIN_ESP" \
    --valid-efd "$VALID_EFD" \
    --valid-esp "$VALID_ESP" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --comparison-name "$COMPARISON_NAME" \
    --equivariance-samples "$EQ_SAMPLES"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Comparison completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Print results location
echo ""
echo "Results saved to: comparisons/$COMPARISON_NAME/"
echo ""
echo "Generated files:"
echo "  - comparison_results.json"
echo "  - performance_comparison.png"
echo "  - efficiency_comparison.png"
echo "  - equivariance_comparison.png"
echo ""

exit $EXIT_CODE

