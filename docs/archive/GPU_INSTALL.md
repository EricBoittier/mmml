# GPU Installation Guide for MMML

This guide specifically covers installing MMML with GPU/CUDA support.

## Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
2. **CUDA Toolkit 12.x** installed
3. **NVIDIA drivers** (version 525.60.13 or higher for CUDA 12)

### Check Your Setup

```bash
# Check GPU and driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Should show CUDA 12.x
```

## Installation Methods

### Method 1: Using Make (Easiest)

The Makefile handles JAX CUDA installation automatically:

```bash
cd /path/to/mmml
make install-gpu
```

This will:
1. Install JAX and jaxlib with CUDA 12 support from the correct index
2. Install CuPy and cuTENSOR
3. Install all other GPU dependencies

### Method 2: Using uv Manually

If you need more control:

```bash
# Step 1: Install JAX with CUDA support from the correct repository
uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]>=0.4.20" "jaxlib[cuda12]>=0.4.20"

# Step 2: Install MMML with GPU extras
uv sync --extra gpu
```

### Method 3: Using Conda

```bash
# Create GPU environment (includes CUDA toolkit from conda)
conda env create -f environment-gpu.yml
conda activate mmml-gpu

# The environment.yml handles JAX CUDA installation automatically
```

### Method 4: Using pip

```bash
# Install JAX with CUDA first
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]" "jaxlib[cuda12]"

# Then install MMML with GPU extras
pip install -e ".[gpu]"
```

## Verification

After installation, verify GPU is detected:

```bash
python verify_install.py
```

You should see:
```
✓ GPU/CUDA                                 Found 1 GPU(s)
```

Or test manually:

```bash
python -c "import jax; print('GPUs:', jax.devices())"
# Should output: GPUs: [cuda(id=0)] or similar
```

## Common Issues and Solutions

### Issue 1: JAX Not Detecting GPU

**Symptom:**
```
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

**Solution:**
```bash
# Uninstall CPU-only JAX
pip uninstall jax jaxlib -y

# Reinstall with CUDA support
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]" "jaxlib[cuda12]"

# Verify
python -c "import jax; print(jax.devices())"
```

### Issue 2: CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA version mismatch
```

**Solution:**
```bash
# Check your CUDA version
nvcc --version

# If you have CUDA 11.x instead of 12.x:
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda11_pip]" "jaxlib[cuda11_pip]"

# Update pyproject.toml gpu extra to use cuda11 instead of cuda12
```

### Issue 3: Out of Memory

**Symptom:**
```
CUDA_ERROR_OUT_OF_MEMORY
```

**Solutions:**

**A. Reduce batch size in your code:**
```python
# Adjust batch size or chunk your data
```

**B. Set XLA memory fraction:**
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Use 70% of GPU memory
```

**C. Enable memory preallocation:**
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Issue 4: CuPy/CuDNN Not Found

**Symptom:**
```
ImportError: libcudnn.so.8: cannot open shared object file
```

**Solution:**

**For conda:**
```bash
conda install -c conda-forge cudnn
```

**For pip:**
```bash
pip install nvidia-cudnn-cu12
```

**Or set library path:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

### Issue 5: Multiple CUDA Versions

**Symptom:**
JAX uses wrong CUDA version.

**Solution:**
```bash
# Specify CUDA path explicitly
export CUDA_HOME=/usr/local/cuda-12.2
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2"
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Then reinstall JAX
pip uninstall jax jaxlib -y
pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]" "jaxlib[cuda12]"
```

## Performance Tips

### 1. Enable XLA Optimizations

```bash
export XLA_FLAGS="--xla_gpu_enable_fast_math=true --xla_gpu_cuda_data_dir=/usr/local/cuda"
```

### 2. Use Mixed Precision (if your model supports it)

```python
import jax
jax.config.update('jax_enable_x64', False)  # Use float32 instead of float64
```

### 3. Profile GPU Usage

```python
import jax
import jax.profiler

# Start profiler
jax.profiler.start_trace("/tmp/tensorboard")

# Your code here

# Stop profiler
jax.profiler.stop_trace()

# View in TensorBoard:
# tensorboard --logdir=/tmp/tensorboard
```

### 4. Check GPU Utilization

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Or use nvtop (more user-friendly)
sudo apt install nvtop
nvtop
```

## Environment Variables Reference

Key environment variables for GPU setups:

```bash
# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.2
export CUDA_PATH=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

# JAX GPU settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

# Visible GPUs (useful for multi-GPU systems)
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
# export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

Add these to your `~/.bashrc` for persistence:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.2' >> ~/.bashrc
echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false' >> ~/.bashrc
source ~/.bashrc
```

## Docker with GPU

For containerized GPU environments:

```bash
# Build GPU image
make docker-build-gpu

# Run with GPU access
make docker-run-gpu

# Or with Docker Compose
docker-compose up -d mmml-gpu
docker-compose exec mmml-gpu bash
```

Docker automatically handles CUDA libraries inside the container.

## Testing GPU Performance

Quick benchmark to verify GPU is working and fast:

```python
import jax
import jax.numpy as jnp
import time

# Create large matrices
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10000, 10000))

# Force compilation and first run
y = jnp.dot(x, x.T)
y.block_until_ready()

# Time it
start = time.time()
y = jnp.dot(x, x.T)
y.block_until_ready()
end = time.time()

print(f"Matrix multiply took: {end - start:.4f} seconds")
print(f"Device: {y.device()}")
print(f"Using: {'GPU' if 'gpu' in str(y.device()).lower() else 'CPU'}")
```

Expected: <0.1 seconds on modern GPU, several seconds on CPU.

## Multi-GPU Setup

For systems with multiple GPUs:

```python
import jax

# See all devices
print("Available devices:", jax.devices())

# Use specific GPU
device = jax.devices('gpu')[0]  # First GPU
with jax.default_device(device):
    # Your code here
    pass

# Parallelize across all GPUs
from jax.experimental import pjit
# See JAX documentation for multi-GPU programming
```

## Getting Help

If you're still having issues:

1. Check CUDA compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
2. Check JAX CUDA compatibility: https://github.com/google/jax#installation
3. Create an issue: https://github.com/EricBoittier/mmml/issues
4. Include output of:
   ```bash
   nvidia-smi
   nvcc --version
   python -c "import jax; print(jax.devices())"
   python verify_install.py
   ```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `make install-gpu` | Install with GPU support |
| `python verify_install.py` | Verify GPU detected |
| `nvidia-smi` | Check GPU status |
| `python -c "import jax; print(jax.devices())"` | Test JAX GPU |
| `export CUDA_VISIBLE_DEVICES=0` | Use specific GPU |
| `watch nvidia-smi` | Monitor GPU usage |

---

**For general installation instructions, see [INSTALL.md](INSTALL.md)**

**For quick start, see [QUICKSTART.md](QUICKSTART.md)**

