#!/bin/bash
#SBATCH --job-name=GPU_Test
#SBATCH --time=00:10:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu6hours
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err

# Quick GPU and environment test
# Purpose: Verify setup before running full training

set -e  # Exit on any error

echo "=========================================="
echo "GPU and Environment Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust as needed for Scicore)
module purge
module load Python/3.11

# Activate conda environment
echo ""
echo "Activating conda environment..."
source activate mmml

# Print environment info
echo ""
echo "=========================================="
echo "Environment Information"
echo "=========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Test JAX
echo "=========================================="
echo "JAX Test"
echo "=========================================="
python << 'EOF'
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Quick computation test
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"✅ JAX computation test passed: {y.shape}")
EOF

# Test e3x
echo ""
echo "=========================================="
echo "e3x Test"
echo "=========================================="
python -c "import e3x; print(f'e3x version: {e3x.__version__}')"

# Test other dependencies
echo ""
echo "=========================================="
echo "Dependencies Test"
echo "=========================================="
python << 'EOF'
import numpy as np
import scipy
import ase
import optax
import flax

print(f"✅ NumPy: {np.__version__}")
print(f"✅ SciPy: {scipy.__version__}")
print(f"✅ ASE: {ase.__version__}")
print(f"✅ Optax: {optax.__version__}")
print(f"✅ Flax: {flax.__version__}")
EOF

# Test data files (if available)
echo ""
echo "=========================================="
echo "Data Files Check"
echo "=========================================="
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

for file in train.npz grids_train.npz valid.npz grids_valid.npz; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✅ Found: $file ($size)"
    else
        echo "⚠️  Not found: $file"
    fi
done

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "You can now run training scripts."
echo ""

