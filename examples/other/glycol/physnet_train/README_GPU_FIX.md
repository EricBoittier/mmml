# GPU/CUDA Initialization Error - Quick Fix Guide

## The Problem You're Seeing

Your training job shows this error:

```
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NOT_INITIALIZED
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

This means **JAX cannot access the GPU** and is running on CPU instead, making training ~10-100x slower.

## What's Happening?

The error `CUDA_ERROR_NOT_INITIALIZED` occurs when:
1. **Most likely:** JAX is not installed with CUDA support, OR
2. **Less likely:** GPU wasn't allocated by SLURM, OR  
3. **Rare:** CUDA version mismatch between system and JAX

## Quick Fix (Try These In Order)

### ✅ Step 1: Test GPU Access

First, verify if the problem is JAX or GPU allocation:

```bash
cd ~/mmml/examples/co2/physnet_train
mkdir -p logs
sbatch test_gpu.sh
```

Wait for the job to complete, then check results:

```bash
# Find your job ID
ls logs/test_gpu_*.out

# Check the output
cat logs/test_gpu_JOBID.out
```

**Look for:**
- ✅ "GPU TEST PASSED" → GPU is working, original job should work now
- ❌ "No GPU device found" → Continue to Step 2
- ❌ "nvidia-smi failed" → Contact cluster admin (GPU not allocated)

### ✅ Step 2: Fix JAX Installation

If Step 1 shows "No GPU device found" but `nvidia-smi` works, reinstall JAX:

```bash
cd ~/mmml/examples/co2/physnet_train
bash fix_jax_cuda.sh
```

This will:
- Uninstall old JAX packages
- Install JAX with CUDA 12 support
- Verify installation

Then **rerun Step 1** to test.

### ✅ Step 3: Rerun Your Training

Once `test_gpu.sh` passes, your environment is ready:

```bash
# Run the quick scan (6 jobs)
sbatch slurm_quick_scan.sh

# Or run the full scan (24 jobs)
sbatch slurm_model_scan.sh
```

## How to Check if Training is Using GPU

Your `.out` log files now include diagnostics. Check for:

```bash
cat logs/scan_JOBID_TASKID.out | grep -A 5 "JAX Installation"
```

**Good (GPU):**
```
JAX devices: [CudaDevice(id=0)]
JAX default backend: gpu
```

**Bad (CPU):**
```
JAX devices: [CpuDevice(id=0)]
JAX default backend: cpu
```

## Training Speed Comparison

- **With GPU:** ~1-5 seconds/epoch → 100 epochs in ~10 minutes
- **With CPU:** ~30-300 seconds/epoch → 100 epochs in ~5-8 hours

If your training is taking hours instead of minutes, you're on CPU!

## Files Created/Updated

1. **test_gpu.sh** - Test script to verify GPU access
2. **fix_jax_cuda.sh** - Script to reinstall JAX with CUDA support
3. **CUDA_TROUBLESHOOTING.md** - Comprehensive troubleshooting guide
4. **slurm_model_scan.sh** - Updated with GPU diagnostics
5. **slurm_quick_scan.sh** - Updated with GPU diagnostics

## Still Not Working?

See **CUDA_TROUBLESHOOTING.md** for:
- Detailed diagnostic steps
- Common causes and solutions
- How to test in interactive sessions
- When to contact cluster admin

## TL;DR

```bash
# 1. Test GPU
sbatch test_gpu.sh
cat logs/test_gpu_*.out

# 2. If test fails, fix JAX
bash fix_jax_cuda.sh

# 3. Test again
sbatch test_gpu.sh
cat logs/test_gpu_*.out

# 4. If test passes, run training
sbatch slurm_quick_scan.sh
```

## Need Help?

When asking for help, provide:
1. Output of `cat logs/test_gpu_*.out`
2. Output of `pip list | grep jax` (in your venv)
3. Output of `module list` (after loading CUDA)
4. Your cluster/partition name (currently: titan)

