#!/bin/bash
# Script to reinstall JAX with proper CUDA 12 support
# Run this if you're getting CUDA_ERROR_NOT_INITIALIZED

set -e

echo "=========================================="
echo "JAX CUDA 12 Reinstallation Script"
echo "=========================================="
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ~/mmml/.venv/bin/activate
fi

echo "Current virtual environment: $VIRTUAL_ENV"
echo ""

# Show current JAX installation
echo "--- Current JAX Installation ---"
pip list | grep -i jax || echo "No JAX packages found"
echo ""

# Confirm with user
echo "This will uninstall and reinstall JAX with CUDA 12 support."
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Uninstall existing JAX packages
echo ""
echo "--- Uninstalling existing JAX packages ---"
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda11-plugin jax-cuda11-pjrt -y || true
echo "Uninstall complete."
echo ""

# Clear pip cache for JAX
echo "--- Clearing pip cache ---"
pip cache remove jax* || true
echo ""

# Install JAX with CUDA 12 support
echo "--- Installing JAX with CUDA 12 support ---"
echo "This may take a few minutes..."
echo ""

# Use the official JAX installation command for CUDA 12
pip install -U "jax[cuda12]"

echo ""
echo "--- Installation Complete ---"
echo ""

# Verify installation
echo "--- Verifying Installation ---"
echo ""
echo "Installed JAX packages:"
pip list | grep -i jax
echo ""

echo "Testing JAX (CPU mode - GPU test requires SLURM job):"
python << 'EOF'
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print()

# Check if CUDA plugin is available (even if no GPU present)
try:
    from jax_plugins import xla_cuda12
    print("✅ CUDA 12 plugin is installed")
except ImportError as e:
    print(f"❌ CUDA 12 plugin not found: {e}")
    print("   But this might be OK if dependencies are correct")

# Simple computation test
x = jnp.ones(10)
result = x.sum()
print(f"\nSimple computation test: {result}")
print("✅ JAX is functional")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ JAX REINSTALLATION SUCCESSFUL"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Test GPU access by submitting: sbatch test_gpu.sh"
    echo "2. Check the output: cat logs/test_gpu_*.out"
    echo "3. If test passes, your environment is ready!"
    echo "4. If test fails, see CUDA_TROUBLESHOOTING.md"
else
    echo ""
    echo "=========================================="
    echo "❌ JAX REINSTALLATION FAILED"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "You may need to manually install JAX:"
    echo "  pip install -U 'jax[cuda12]'"
    exit 1
fi

