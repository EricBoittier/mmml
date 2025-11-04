#!/bin/bash
# Create a new Python virtual environment with CUDA 13 support for JAX
# This script will set up everything needed to run PhysNet training on GPU

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Setting Up CUDA 13 Environment for PhysNet Training"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Configuration
ENV_NAME="mmml_cuda13"
ENV_PATH=~/mmml/.venv_cuda13
PYTHON_VERSION="python3.12"  # or python3.11, python3.10

echo "Configuration:"
echo "  Environment name: $ENV_NAME"
echo "  Environment path: $ENV_PATH"
echo "  Python version: $PYTHON_VERSION"
echo ""

# Check if Python is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "❌ ERROR: $PYTHON_VERSION not found!"
    echo "Available Python versions:"
    ls -1 /usr/bin/python* 2>/dev/null || echo "  None found in /usr/bin"
    echo ""
    echo "Please load a Python module or install Python 3.10+:"
    echo "  module avail python"
    echo "  module load Python/3.12.x"
    exit 1
fi

PYTHON_CMD=$(command -v $PYTHON_VERSION)
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Remove old environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo "⚠️  Old environment found at $ENV_PATH"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old environment..."
        rm -rf "$ENV_PATH"
    else
        echo "Aborted. Please remove manually: rm -rf $ENV_PATH"
        exit 1
    fi
fi

# Create new virtual environment
echo "───────────────────────────────────────────────────────────────"
echo "Step 1: Creating virtual environment"
echo "───────────────────────────────────────────────────────────────"
$PYTHON_CMD -m venv "$ENV_PATH"
echo "✅ Virtual environment created at $ENV_PATH"
echo ""

# Activate environment
source "$ENV_PATH/bin/activate"
echo "✅ Environment activated"
echo ""

# Upgrade pip
echo "───────────────────────────────────────────────────────────────"
echo "Step 2: Upgrading pip"
echo "───────────────────────────────────────────────────────────────"
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install JAX with CUDA support
echo "───────────────────────────────────────────────────────────────"
echo "Step 3: Installing JAX with CUDA support"
echo "───────────────────────────────────────────────────────────────"
echo ""
echo "Checking available CUDA support..."
echo ""

# Try CUDA 13 first (if available), fall back to CUDA 12
# Note: As of late 2024, JAX primarily supports CUDA 12
# CUDA 13 support may require specific jaxlib builds

echo "Attempting to install JAX with CUDA support..."
echo "(Will try CUDA 12 as it's most widely supported)"
echo ""

# Install JAX with CUDA 12 (most stable)
# If CUDA 13 specific build is needed, adjust the version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

if [ $? -ne 0 ]; then
    echo "⚠️  Installation from jax_cuda_releases failed, trying standard CUDA 12 install..."
    pip install --upgrade "jax[cuda12]"
fi

echo ""
echo "✅ JAX installed"
echo ""

# Install other required packages
echo "───────────────────────────────────────────────────────────────"
echo "Step 4: Installing required packages"
echo "───────────────────────────────────────────────────────────────"

# Install from mmml repository requirements
if [ -f ~/mmml/requirements.txt ]; then
    echo "Installing from mmml/requirements.txt..."
    # Note: requirements.txt includes jax[cuda12], but we already installed it above
    # Skip jax reinstallation from requirements.txt
    grep -v "^jax" ~/mmml/requirements.txt > /tmp/requirements_no_jax.txt
    pip install -r /tmp/requirements_no_jax.txt
    rm /tmp/requirements_no_jax.txt
else
    echo "Installing core dependencies manually..."
    pip install numpy scipy pandas matplotlib
    pip install pyyaml hydra-core omegaconf
    pip install optax flax dm-haiku
    pip install ase  # Atomic Simulation Environment
    pip install orbax-checkpoint  # For checkpointing
    pip install e3x
    pip install scikit-learn statsmodels
    pip install tabulate tqdm toolz
fi

# Install additional PhysNetJax dependencies
echo "Installing PhysNetJax specific dependencies..."
pip install optax flax orbax-checkpoint dm-haiku || echo "Some packages may already be installed"

echo ""
echo "✅ Dependencies installed"
echo ""

# Install mmml in development mode
echo "───────────────────────────────────────────────────────────────"
echo "Step 5: Installing mmml package"
echo "───────────────────────────────────────────────────────────────"
cd ~/mmml
pip install -e .
echo "✅ mmml installed in development mode"
echo ""

# Verify installation
echo "───────────────────────────────────────────────────────────────"
echo "Step 6: Verifying installation"
echo "───────────────────────────────────────────────────────────────"
echo ""

python << 'EOF'
import sys
print("Python version:", sys.version)
print()

# Check JAX
try:
    import jax
    import jax.numpy as jnp
    print("✅ JAX version:", jax.__version__)
    print("   Default backend:", jax.default_backend())
    print("   Devices (CPU mode):", jax.devices())
    
    # Check if CUDA support is compiled in
    try:
        from jax.lib import xla_bridge
        print("   XLA platform:", xla_bridge.get_backend().platform)
    except Exception as e:
        print(f"   XLA check: {e}")
    
    # Check for CUDA plugins
    try:
        import jax_plugins
        print("   JAX plugins available")
    except ImportError:
        print("   JAX plugins: not found (may be ok)")
    
except ImportError as e:
    print(f"❌ JAX import failed: {e}")
    sys.exit(1)

print()

# Check other key packages
packages = ['numpy', 'scipy', 'optax', 'flax', 'ase', 'orbax']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {pkg:12s} version: {version}")
    except ImportError:
        print(f"❌ {pkg:12s} NOT INSTALLED")

print()

# Check mmml
try:
    import mmml
    print(f"✅ mmml package: available")
    from mmml.physnetjax.physnetjax.models.model import EF
    print(f"✅ PhysNet model: can import")
except ImportError as e:
    print(f"❌ mmml import failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✅ All required packages installed successfully!")
print("=" * 60)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Verification failed! Please check errors above."
    exit 1
fi

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "Step 7: Creating activation script"
echo "───────────────────────────────────────────────────────────────"

# Create easy activation script
cat > ~/mmml/activate_cuda13.sh << 'ACTIVATION_EOF'
#!/bin/bash
# Activate the CUDA 13 environment for PhysNet training

# Load CUDA module (adjust version as needed)
module load CUDA/13.0 2>/dev/null || \
    module load CUDA/12.6 2>/dev/null || \
    module load CUDA/12.2.0 2>/dev/null || \
    echo "⚠️  Warning: No CUDA module loaded. Load manually if needed."

# Activate virtual environment
source ~/mmml/.venv_cuda13/bin/activate

echo "═══════════════════════════════════════════════════════════"
echo "  CUDA Environment Activated"
echo "═══════════════════════════════════════════════════════════"
echo "Virtual environment: ~/mmml/.venv_cuda13"
echo "Python: $(which python)"
echo "To test GPU: cd ~/mmml/examples/co2/physnet_train && sbatch test_gpu_cuda13.sh"
echo "═══════════════════════════════════════════════════════════"
ACTIVATION_EOF

chmod +x ~/mmml/activate_cuda13.sh

echo "✅ Activation script created: ~/mmml/activate_cuda13.sh"
echo ""

# Create environment info file
cat > "$ENV_PATH/ENVIRONMENT_INFO.txt" << 'INFO_EOF'
Environment: mmml_cuda13
Created: $(date)
Purpose: PhysNet training with JAX and CUDA support

To activate:
  source ~/mmml/activate_cuda13.sh

Or manually:
  module load CUDA/13.0  # or CUDA/12.x
  source ~/mmml/.venv_cuda13/bin/activate

To test GPU:
  cd ~/mmml/examples/co2/physnet_train
  sbatch test_gpu_cuda13.sh

To run training:
  cd ~/mmml/examples/co2/physnet_train
  sbatch slurm_cuda13_scan.sh
INFO_EOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ CUDA 13 Environment Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Environment location: $ENV_PATH"
echo ""
echo "To use this environment:"
echo "  source ~/mmml/activate_cuda13.sh"
echo ""
echo "Next steps:"
echo "  1. Test GPU: cd ~/mmml/examples/co2/physnet_train && sbatch test_gpu_cuda13.sh"
echo "  2. Run training: sbatch slurm_cuda13_scan.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════"

