#!/bin/bash
# Sweep number of features: [16, 32, 64, 128]

#SBATCH --job-name=sweep_features
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_features_%A_%a.out
#SBATCH --error=logs/sweep_features_%A_%a.err

set -e

echo "=========================================="
echo "Sweep: features"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# ---- Environment ----
module load CUDA/12.2.0

echo "--- GPU Information ---"
nvidia-smi || echo "nvidia-smi failed"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

source ~/mmml/.venv/bin/activate

python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')" \
    || echo "JAX check failed"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- Parameter sweep ----
FEATURES_LIST=(16 32 64 128)
FEATURES=${FEATURES_LIST[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "Configuration:"
echo "  features:            $FEATURES"
echo "  max_degree:          4  (fixed)"
echo "  num_basis_functions: 32 (fixed)"
echo "  num_iterations:      2  (fixed)"
echo ""

python -u training.py \
    --data data-full.npz \
    --features "$FEATURES" \
    --max_degree 4 \
    --num_basis_functions 32 \
    --num_iterations 2 \
    --cutoff 10.0 \
    --num_train 8000 \
    --num_valid 1000 \
    --num_epochs 500 \
    --learning_rate 0.0001 \
    --batch_size 10 \
    --clip_norm 1000.0 \
    --ema_decay 0.9 \
    --energy_weight 1.0 \
    --forces_weight 1000.0 \
    --dipole_weight 20.0 \
    --reduce_on_plateau_patience 5 \
    --reduce_on_plateau_cooldown 5 \
    --reduce_on_plateau_factor 0.9 \
    --reduce_on_plateau_rtol 1e-4 \
    --reduce_on_plateau_accumulation_size 5 \
    --reduce_on_plateau_min_scale 0.01

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "features=$FEATURES"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
