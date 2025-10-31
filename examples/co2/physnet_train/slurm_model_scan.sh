#!/bin/bash

#SBATCH --job-name=PhysNet_Scan
#SBATCH --time=06:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --array=0-23  # 24 different model configurations
#SBATCH --cpus-per-task=2
#SBATCH --partition=titan  # or a100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/scan_%A_%a.out
#SBATCH --error=logs/scan_%A_%a.err

# Load required modules
module load CUDA/12.2.0

# Activate virtual environment
source ~/mmml/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Get array task ID
X=$SLURM_ARRAY_TASK_ID
echo "Running configuration $X"

# Define hyperparameter combinations
# Format: FEATURES|NUM_ITERATIONS|NUM_BASIS|N_RES|BATCH_SIZE|MAX_DEGREE
declare -a CONFIGS=(
    # Small models - scalar only (max_degree=0)
    "32|3|32|2|32|0"
    "32|4|32|2|32|0"
    "32|5|32|3|32|0"
    "64|3|32|2|32|0"
    "64|4|32|3|32|0"
    "64|5|32|3|32|0"
    
    # Medium models - scalar and vectors (max_degree=0,2)
    "64|3|64|2|16|0"
    "64|4|64|3|16|0"
    "64|5|64|3|16|2"
    "128|3|64|2|16|0"
    "128|4|64|3|16|0"
    "128|5|64|3|16|2"
    
    # Medium-large models - with vectors
    "128|3|64|3|8|0"
    "128|4|64|3|8|2"
    "128|5|64|4|8|2"
    "256|3|64|2|8|0"
    "256|4|64|3|8|2"
    "256|5|64|3|8|2"
    
    # Large models - with higher degree
    "128|4|128|3|8|2"
    "128|5|128|4|8|2"
    "256|3|128|3|4|2"
    "256|4|128|3|4|3"
    "256|5|128|4|4|3"
    "512|3|128|3|4|2"
)

# Get configuration for this array task
CONFIG=${CONFIGS[$X]}
IFS='|' read -r FEATURES ITERATIONS BASIS N_RES BATCH MAX_DEGREE <<< "$CONFIG"

echo "Configuration:"
echo "  Features: $FEATURES"
echo "  Iterations: $ITERATIONS"
echo "  Basis functions: $BASIS"
echo "  Residual blocks: $N_RES"
echo "  Batch size: $BATCH"
echo "  Max degree: $MAX_DEGREE"

# Create experiment name
EXP_NAME="co2_scan_f${FEATURES}_i${ITERATIONS}_b${BASIS}_r${N_RES}_bs${BATCH}_d${MAX_DEGREE}"

echo "Experiment: $EXP_NAME"
echo "Starting training..."

# Run training
python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --name "$EXP_NAME" \
  --features $FEATURES \
  --max-degree $MAX_DEGREE \
  --num-iterations $ITERATIONS \
  --num-basis-functions $BASIS \
  --n-res $N_RES \
  --batch-size $BATCH \
  --natoms 3 \
  --epochs 500 \
  --learning-rate 0.001 \
  --schedule warmup_cosine \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --energy-unit eV \
  --subtract-atomic-energies \
  --atomic-energy-method default \
  --no-energy-bias \
  --cutoff 6.0 \
  --print-freq 10 \
  --save-best \
  --objective valid_forces_mae \
  --verbose

echo "Training complete for $EXP_NAME"

