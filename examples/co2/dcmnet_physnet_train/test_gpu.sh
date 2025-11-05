#!/bin/bash
#SBATCH --job-name=GPU_Test
<<<<<<< HEAD
#SBATCH --time=1-00:00:00
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu1day
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --array=0-10
=======
#SBATCH --time=00:10:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu6hours
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
>>>>>>> f5899bcf5dd3cccc2bb1f96f6f269611600b038b

echo "=========================================="
echo "GPU Test Script"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

<<<<<<< HEAD
module activate Miniconda3
conda init bash
conda activate mmml-gpu

export NDCM=1

export seed=$SLURM_JOB_ID
echo $seed 
RunPython="/scicore/home/meuwly/boitti0000/.conda/envs/mmml-gpu/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" && $RunPython compare_models.py \
    --train-efd energies_forces_dipoles_train.npz \
    --train-esp grids_esp_train.npz \
    --valid-efd energies_forces_dipoles_valid.npz \
    --valid-esp grids_esp_valid.npz \
    --epochs 10 \
    --n-dcm $NDCM \
    --batch-size 100 \
    --comparison-name dcm1test$seed \
    --seed $seed
=======
# Load CUDA
echo "Loading CUDA module..."
module load CUDA/12.2.0
echo "Module loaded."
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
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check nvcc
echo "--- CUDA Compiler ---"
which nvcc && nvcc --version || echo "nvcc not found in PATH"
echo ""

# Activate virtual environment
echo "--- Activating Virtual Environment ---"
source ~/mmml/.venv/bin/activate
echo "Virtual environment activated."
echo ""

# Check Python packages
echo "--- Python Environment ---"
echo "Python: $(which python)"
python --version
echo ""
echo "JAX packages:"
pip list | grep -i jax
echo ""

# Test JAX
echo "--- JAX GPU Test ---"
python << 'EOF'
import sys
import jax
import jax.numpy as jnp

print("JAX Configuration:")
print(f"  Version: {jax.__version__}")
print(f"  Default backend: {jax.default_backend()}")
print(f"  Available devices: {jax.devices()}")
print()

# Check if GPU is available
devices = jax.devices()
has_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)

if not has_gpu:
    print("❌ ERROR: No GPU device found!")
    print("   JAX is using CPU only.")
    print()
    print("Likely causes:")
    print("  1. JAX not installed with CUDA support")
    print("  2. CUDA version mismatch")
    print("  3. GPU not visible to the job")
    sys.exit(1)

print("✅ GPU device found!")
print()

# Test computation
print("Testing computation on GPU...")
try:
    # Create array on GPU
    x = jnp.arange(10000).reshape(100, 100)
    print(f"  Created array with shape {x.shape}")
    print(f"  Array device: {x.devices()}")
    
    # Perform computation
    result = jnp.dot(x, x.T).sum()
    print(f"  Matrix computation result: {result}")
    
    # Force computation to complete
    result.block_until_ready()
    
    print()
    print("✅ GPU computation successful!")
    print()
    print("=" * 50)
    print("GPU TEST PASSED - JAX is working correctly!")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ ERROR during computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ALL TESTS PASSED"
    echo "Your environment is ready for GPU training!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ TESTS FAILED"
    echo "Please check the errors above."
    echo "See CUDA_TROUBLESHOOTING.md for solutions."
    echo "=========================================="
    exit 1
fi
>>>>>>> f5899bcf5dd3cccc2bb1f96f6f269611600b038b

