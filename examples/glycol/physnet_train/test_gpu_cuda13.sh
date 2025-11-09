#!/bin/bash
#SBATCH --job-name=GPU_Test_CUDA13
#SBATCH --time=00:10:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu6hours
#SBATCH --output=logs/test_gpu_cuda13_%j.out
#SBATCH --error=logs/test_gpu_cuda13_%j.err

echo "=========================================="
echo "GPU Test Script - CUDA 13 Environment"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Try to load CUDA 13, fall back to available versions
echo "Loading CUDA module..."
if module load CUDA/13.0 2>/dev/null; then
    echo "✅ Loaded CUDA 13.0"
elif module load CUDA/12.6 2>/dev/null; then
    echo "✅ Loaded CUDA 12.6 (13.0 not available)"
elif module load CUDA/12.3 2>/dev/null; then
    echo "✅ Loaded CUDA 12.3 (13.0 not available)"
elif module load CUDA/12.2.0 2>/dev/null; then
    echo "✅ Loaded CUDA 12.2.0 (13.0 not available)"
else
    echo "❌ ERROR: Could not load any CUDA module!"
    echo "Available CUDA modules:"
    module avail CUDA
    exit 1
fi
echo ""

# Check GPU
echo "--- GPU Check with nvidia-smi ---"
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed!"
    echo "GPU may not be allocated to this job."
    exit 1
fi
echo ""

# Check environment
echo "--- CUDA Environment Variables ---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check nvcc
echo "--- CUDA Compiler ---"
which nvcc && nvcc --version || echo "nvcc not found in PATH"
echo ""

# Check loaded modules
echo "--- Loaded Modules ---"
module list
echo ""

# Activate CUDA 13 virtual environment
echo "--- Activating CUDA 13 Virtual Environment ---"
if [ -f ~/mmml/.venv_cuda13/bin/activate ]; then
    source ~/mmml/.venv_cuda13/bin/activate
    echo "✅ Virtual environment activated: ~/mmml/.venv_cuda13"
else
    echo "❌ ERROR: CUDA 13 environment not found!"
    echo "Please run: bash setup_cuda13_env.sh"
    exit 1
fi
echo ""

# Check Python packages
echo "--- Python Environment ---"
echo "Python: $(which python)"
python --version
echo ""
echo "JAX packages:"
pip list | grep -i jax
echo ""
echo "Other key packages:"
pip list | grep -E 'numpy|optax|flax|orbax'
echo ""

# Test JAX
echo "=========================================="
echo "JAX GPU Test"
echo "=========================================="
python << 'EOF'
import sys
import jax
import jax.numpy as jnp

print("JAX Configuration:")
print(f"  Version: {jax.__version__}")
print(f"  Default backend: {jax.default_backend()}")
print(f"  Available devices: {jax.devices()}")

# Check XLA backend
try:
    from jax.lib import xla_bridge
    print(f"  XLA platform: {xla_bridge.get_backend().platform}")
    print(f"  XLA client: {xla_bridge.get_backend().client}")
except Exception as e:
    print(f"  XLA info: {e}")

print()

# Check if GPU is available
devices = jax.devices()
has_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)

if not has_gpu:
    print("❌ ERROR: No GPU device found!")
    print("   JAX is using CPU only.")
    print()
    print("Possible causes:")
    print("  1. JAX not installed with CUDA support")
    print("  2. CUDA version mismatch")
    print("  3. GPU not visible to the job")
    print()
    print("To fix:")
    print("  cd ~/mmml/examples/co2/physnet_train")
    print("  bash setup_cuda13_env.sh")
    sys.exit(1)

print("✅ GPU device found!")
print()

# Test computation
print("Testing computation on GPU...")
try:
    # Create array on GPU
    print("  Creating test arrays...")
    x = jnp.arange(10000).reshape(100, 100)
    print(f"  Array shape: {x.shape}")
    print(f"  Array device: {x.devices()}")
    
    # Perform computation
    print("  Running matrix multiplication...")
    result = jnp.dot(x, x.T).sum()
    print(f"  Computation result: {result}")
    
    # Force computation to complete
    result.block_until_ready()
    
    # More complex test
    print("  Running more complex operations...")
    y = jnp.sin(x) + jnp.cos(x)
    z = jnp.exp(y * 0.01).mean()
    z.block_until_ready()
    print(f"  Complex operation result: {z}")
    
    print()
    print("✅ GPU computation successful!")
    print()
    
except Exception as e:
    print(f"❌ ERROR during computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test mmml imports
print("Testing mmml package imports...")
try:
    from mmml.physnetjax.physnetjax.models.model import EF
    print("✅ PhysNet model import successful")
    
    from mmml.physnetjax.physnetjax.training.training import train_model
    print("✅ Training functions import successful")
    
    from mmml.data import load_npz, DataConfig
    print("✅ Data loading functions import successful")
    
except ImportError as e:
    print(f"❌ mmml import failed: {e}")
    print()
    print("mmml may not be installed in this environment.")
    print("The setup script should have installed it.")
    sys.exit(1)

print()
print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("Your CUDA 13 environment is ready for training!")
print()
print("Next step: Run training with")
print("  sbatch slurm_cuda13_scan.sh")
print("=" * 60)

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ GPU TEST PASSED"
    echo "=========================================="
    echo ""
    echo "Your CUDA 13 environment is working correctly!"
    echo "You can now run training jobs."
    echo ""
    echo "To run training:"
    echo "  sbatch slurm_cuda13_scan.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ GPU TEST FAILED"
    echo "=========================================="
    echo ""
    echo "Please check the errors above."
    echo "You may need to recreate the environment:"
    echo "  bash setup_cuda13_env.sh"
    echo ""
    exit 1
fi

