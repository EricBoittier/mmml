# CUDA Initialization Error Troubleshooting Guide

## The Problem

You're encountering this error:
```
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NOT_INITIALIZED
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

This means JAX cannot initialize CUDA, causing training to run on CPU instead of GPU.

## Diagnostic Steps

### 1. Check GPU Allocation

The updated SLURM scripts now include diagnostics. When you run your job, check the `.out` log file for:

```bash
# Look for GPU information
cat logs/scan_<JOBID>_<TASKID>.out | grep -A 20 "GPU and CUDA Diagnostics"
```

**What to look for:**
- Does `nvidia-smi` show a GPU?
- Is `CUDA_VISIBLE_DEVICES` set? (should be `0` or similar, not empty)
- What does JAX report for devices?

### 2. Verify JAX CUDA Installation

In your virtual environment, check JAX version:

```bash
source ~/mmml/.venv/bin/activate
pip list | grep jax
```

**Expected output:**
```
jax                    0.4.x
jaxlib                 0.4.x+cuda12.cudnn89  # <-- Should have cuda12
jax-cuda12-plugin      0.x.x
```

**Problem signs:**
- If you see just `jaxlib` without `+cuda12`, you have CPU-only JAX
- If you see `+cuda11` but system has CUDA 12, version mismatch

### 3. Test JAX GPU Access

Run this test in an interactive SLURM session with GPU:

```bash
srun --partition=titan --gres=gpu:1 --pty bash
module load CUDA/12.2.0
source ~/mmml/.venv/bin/activate

python -c "
import jax
print('JAX version:', jax.__version__)
print('Available devices:', jax.devices())
print('Default backend:', jax.default_backend())

# Test a simple computation
import jax.numpy as jnp
x = jnp.ones(10)
print('Test computation successful:', x.sum())
"
```

**Expected output:**
```
JAX version: 0.4.x
Available devices: [CudaDevice(id=0)]  # <-- Should be CudaDevice, not CpuDevice
Default backend: gpu
Test computation successful: 10.0
```

## Common Causes & Fixes

### Cause 1: JAX Not Installed with CUDA Support

**Symptom:** `jaxlib` without `+cuda12` suffix

**Fix:** Reinstall JAX with CUDA 12 support:

```bash
source ~/mmml/.venv/bin/activate
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt -y

# Install JAX with CUDA 12 support
pip install -U "jax[cuda12]"
```

### Cause 2: CUDA Version Mismatch

**Symptom:** System has CUDA 12.2 but JAX installed for CUDA 11

**Fix:** Same as above - reinstall with correct CUDA version

### Cause 3: GPU Not Allocated by SLURM

**Symptom:** 
- `nvidia-smi` fails or shows no processes
- `CUDA_VISIBLE_DEVICES` is empty

**Fix:** Check SLURM configuration:
```bash
# Verify GPU is requested in script
grep "gres=gpu" slurm_model_scan.sh  # Should show: #SBATCH --gres=gpu:1

# Check if titan partition has GPUs
sinfo -p titan -o "%P %G %N"

# Try different partition or QoS
```

### Cause 4: Module Load Failure

**Symptom:** CUDA module loaded but environment variables not set

**Fix:** Add explicit environment setup:

```bash
# After module load, manually set if needed:
export CUDA_HOME=/path/to/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### Cause 5: Conflicting CUDA Libraries

**Symptom:** Multiple CUDA versions in environment

**Fix:** Clean environment:
```bash
module purge
module load CUDA/12.2.0
# Then activate venv and check LD_LIBRARY_PATH
```

## Solution Checklist

Run through these in order:

- [ ] **Step 1:** Run updated SLURM script with diagnostics
- [ ] **Step 2:** Check `.out` log file - does `nvidia-smi` work?
- [ ] **Step 3:** Check if `CUDA_VISIBLE_DEVICES` is set
- [ ] **Step 4:** Verify JAX reports GPU devices
- [ ] **Step 5:** If GPU missing, contact cluster admin about node/partition
- [ ] **Step 6:** If GPU present but JAX fails, reinstall JAX with CUDA 12
- [ ] **Step 7:** Test in interactive session before resubmitting jobs

## Quick Test Script

Save this as `test_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=GPU_Test
#SBATCH --time=00:10:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --output=test_gpu.out
#SBATCH --error=test_gpu.err

module load CUDA/12.2.0
source ~/mmml/.venv/bin/activate

echo "=== GPU Test ==="
nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
python -c "
import jax
import jax.numpy as jnp
print('JAX devices:', jax.devices())
print('Computing on:', jnp.ones(10).devices())
x = jnp.arange(1000).reshape(10, 100)
result = jnp.dot(x, x.T).sum()
print('Computation result:', result)
print('Test PASSED!')
"
```

Run with: `sbatch test_gpu.sh`

## Expected Training Behavior

**With GPU (correct):**
- Training starts immediately
- Epoch time: ~1-5 seconds per epoch (depending on model size)
- No CUDA warnings

**With CPU (problem):**
- Long initialization time
- Epoch time: ~30-300 seconds per epoch
- Warning: "Falling back to cpu"

## Getting Help

If none of these solutions work:

1. Share the output from the diagnostic section in your `.out` log
2. Share the output of: `pip list | grep jax`
3. Share the output of: `module list`
4. Contact your cluster administrator about GPU access on the titan partition

## Additional Resources

- JAX GPU Installation: https://jax.readthedocs.io/en/latest/installation.html
- JAX FAQ: https://jax.readthedocs.io/en/latest/faq.html#gpu-and-tpu
- Check cluster documentation for GPU job submission guidelines

