#!/bin/bash
# Setup script for creating the mmml conda environment on Scicore
# Run this ONCE on Scicore to set up your environment

set -e

echo "=========================================="
echo "MMML Environment Setup for Scicore"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    echo "Please load conda module first:"
    echo "  module load Anaconda3"
    echo "or"
    echo "  module load Miniconda3"
    exit 1
fi

echo "✅ Found conda: $(which conda)"
echo ""

# Determine repo root (3 levels up from sbatch directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "Repository root: $REPO_ROOT"
echo ""

# Choose between CPU and GPU environment
echo "Choose environment type:"
echo "  1) GPU (recommended for Scicore with GPU partition)"
echo "  2) CPU (for testing or CPU-only nodes)"
read -p "Enter choice [1]: " ENV_CHOICE
ENV_CHOICE=${ENV_CHOICE:-1}

if [ "$ENV_CHOICE" = "1" ]; then
    ENV_FILE="environment-gpu.yml"
    echo "Using GPU environment"
elif [ "$ENV_CHOICE" = "2" ]; then
    ENV_FILE="environment.yml"
    echo "Using CPU environment"
else
    echo "Invalid choice, using GPU environment"
    ENV_FILE="environment-gpu.yml"
fi

echo ""

# Check if environment file exists
if [ ! -f "$REPO_ROOT/$ENV_FILE" ]; then
    echo "❌ Error: Environment file not found: $REPO_ROOT/$ENV_FILE"
    exit 1
fi

echo "Environment file: $REPO_ROOT/$ENV_FILE"
echo ""

# Ask for confirmation
read -p "Create 'mmml' conda environment? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Creating conda environment..."
echo "=========================================="
echo ""

# Create environment
cd "$REPO_ROOT"
conda env create -f "$ENV_FILE"

echo ""
echo "=========================================="
echo "Environment created successfully!"
echo "=========================================="
echo ""

# Verify environment
echo "Verifying environment..."
source activate mmml

echo "Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "Testing key packages..."
python << 'EOF'
try:
    import jax
    print(f"✅ JAX {jax.__version__}")
except ImportError as e:
    print(f"❌ JAX import failed: {e}")

try:
    import e3x
    print(f"✅ e3x {e3x.__version__}")
except ImportError as e:
    print(f"❌ e3x import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import ase
    print(f"✅ ASE {ase.__version__}")
except ImportError as e:
    print(f"❌ ASE import failed: {e}")

try:
    import optax
    print(f"✅ Optax {optax.__version__}")
except ImportError as e:
    print(f"❌ Optax import failed: {e}")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  source activate mmml"
echo ""
echo "Or add to your .bashrc:"
echo "  echo 'alias activate_mmml=\"source activate mmml\"' >> ~/.bashrc"
echo ""
echo "Next steps:"
echo "  1. Test GPU access: sbatch 05_gpu_test.sbatch"
echo "  2. Edit data paths in sbatch scripts"
echo "  3. Run training: sbatch 01_train_dcmnet_quick.sbatch"
echo ""

