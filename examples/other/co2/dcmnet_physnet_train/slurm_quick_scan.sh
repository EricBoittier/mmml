#!/bin/bash
# Smaller hyperparameter scan - just 6 configs focusing on model features

#SBATCH --job-name=PhysNet_Quick
#SBATCH --time=06:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --array=0-5  # 6 different model sizes
#SBATCH --cpus-per-task=2
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --output=logs/quick_%A_%a.out
#SBATCH --error=logs/quick_%A_%a.err

module load CUDA/12.2.0

# ============ GPU DIAGNOSTICS ============
echo "=========================================="
echo "GPU and CUDA Diagnostics"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo ""
echo "--- GPU Information ---"
nvidia-smi || echo "ERROR: nvidia-smi failed"
echo ""
echo "--- CUDA Environment ---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which nvcc && nvcc --version || echo "nvcc not found"
echo ""
echo "--- Module Information ---"
module list
echo "=========================================="
echo ""

source ~/mmml/.venv/bin/activate

# Check JAX installation
echo "--- JAX Installation ---"
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}'); print(f'JAX default backend: {jax.default_backend()}')" || echo "ERROR: JAX check failed"
echo "=========================================="
echo ""

mkdir -p logs

X=$SLURM_ARRAY_TASK_ID
echo "Running quick scan configuration $X"

# Simple scan: vary features, iterations, and max_degree
# Format: FEATURES|ITERATIONS|MAX_DEGREE
declare -a CONFIGS=(
    "32|2|2"    # Tiny, scalar only
    "64|2|2"    # Small, scalar only
    "128|4|0"   # Medium, scalar only
    "256|4|0"   # Large, scalar only
    "128|3|2"   # Medium-deep with vectors
    "256|3|2"   # Large-deep with vectors
)

CONFIG=${CONFIGS[$X]}
IFS='|' read -r FEATURES ITERATIONS MAX_DEGREE <<< "$CONFIG"

EXP_NAME="co2_quick_f${FEATURES}_i${ITERATIONS}_d${MAX_DEGREE}"

echo "Features: $FEATURES, Iterations: $ITERATIONS, Max Degree: $MAX_DEGREE"
echo "Running: $EXP_NAME"

# Use unbuffered Python output for real-time SLURM logging
python -u trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --name "$EXP_NAME" \
  --features $FEATURES \
  --max-degree $MAX_DEGREE \
  --num-iterations $ITERATIONS \
  --num-basis-functions 64 \
  --n-res 3 \
  --batch-size 16 \
  --natoms 3 \
  --epochs 100 \
  --learning-rate 0.001 \
  --schedule constant \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --energy-unit eV \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression \
  --no-energy-bias \
  --save-best \
  --objective valid_forces_mae \
  --verbose

echo "Complete: $EXP_NAME"

