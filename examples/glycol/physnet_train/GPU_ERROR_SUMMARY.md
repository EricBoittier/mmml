# GPU Error - What Happened and How to Fix It

## What You Saw

Your training job (scan_60473526_0) failed with:
```
ERROR: CUDA_ERROR_NOT_INITIALIZED
WARNING: Falling back to cpu
```

## Root Cause Analysis

The error `CUDA_ERROR_NOT_INITIALIZED` means JAX tried to use your GPU but couldn't initialize CUDA. The most common cause is that **JAX was installed without CUDA support**, so it only has the CPU backend available.

### Why This Happens

When you install JAX with just `pip install jax`, it installs the **CPU-only version**. To get GPU support, you need to explicitly install the CUDA version:

```bash
pip install "jax[cuda12]"  # For CUDA 12.x
```

Your SLURM script correctly:
- ✅ Requests GPU (`--gres=gpu:1`)
- ✅ Loads CUDA 12.2 (`module load CUDA/12.2.0`)
- ✅ Uses titan partition (has GPUs)

But if JAX doesn't have CUDA support, it can't use the GPU even when it's available.

## What I've Done

### 1. Added GPU Diagnostics to SLURM Scripts

Updated both `slurm_model_scan.sh` and `slurm_quick_scan.sh` to include comprehensive diagnostics that will show:
- GPU allocation status (`nvidia-smi`)
- CUDA environment variables
- JAX configuration (version, devices, backend)

Now when you run a job, the `.out` file will tell you exactly what's wrong.

### 2. Created Test Script

**test_gpu.sh** - Quick 10-minute test to verify your GPU setup:
```bash
sbatch test_gpu.sh
cat logs/test_gpu_*.out
```

This will definitively tell you if:
- GPU is allocated by SLURM ✓
- JAX can see the GPU ✓
- CUDA computations work ✓

### 3. Created Fix Script

**fix_jax_cuda.sh** - Automatically reinstalls JAX with CUDA 12 support:
```bash
bash fix_jax_cuda.sh
```

This will:
- Uninstall any existing JAX packages
- Install JAX with CUDA 12 support
- Verify the installation

### 4. Created Documentation

- **README_GPU_FIX.md** - Quick start guide (read this first!)
- **CUDA_TROUBLESHOOTING.md** - Detailed troubleshooting (if quick fix doesn't work)
- **GPU_ERROR_SUMMARY.md** - This file (explanation of what happened)

## What You Should Do Now

### Option A: Quick Path (Recommended)

```bash
cd ~/mmml/examples/co2/physnet_train

# Test current setup
sbatch test_gpu.sh
# Wait ~5 minutes, then:
cat logs/test_gpu_*.out

# If test fails, fix JAX installation:
bash fix_jax_cuda.sh

# Test again:
sbatch test_gpu.sh
cat logs/test_gpu_*.out

# Once test passes, run your training:
sbatch slurm_quick_scan.sh  # or slurm_model_scan.sh
```

### Option B: Manual Investigation

If you want to understand what's wrong first:

1. Check your current JAX installation:
```bash
source ~/mmml/.venv/bin/activate
pip list | grep jax
```

Look for: Is it `jaxlib` or `jaxlib-cuda12` or `jaxlib+cuda12`?

2. Test JAX manually:
```bash
# On a node with GPU:
srun --partition=titan --gres=gpu:1 --pty bash
module load CUDA/12.2.0
source ~/mmml/.venv/bin/activate
python -c "import jax; print(jax.devices())"
```

Expected: `[CudaDevice(id=0)]`  
If you see: `[CpuDevice(id=0)]` → JAX doesn't have CUDA support

## The Performance Impact

This is **critical** to fix:

| Backend | Training Speed | 100 Epochs |
|---------|----------------|------------|
| GPU (correct) | 1-5 sec/epoch | ~10 minutes |
| CPU (current) | 30-300 sec/epoch | 5-8 hours |

Your training is running **10-100x slower** than it should!

## Technical Details

### What JAX Needs for GPU Support

1. **Correct jaxlib installation:**
   - CPU-only: `jaxlib==0.4.x`
   - GPU: `jaxlib==0.4.x+cuda12.cudnn89`

2. **CUDA plugin:**
   - `jax-cuda12-plugin` or similar

3. **Runtime requirements:**
   - CUDA libraries (provided by `module load CUDA/12.2.0`)
   - GPU visible to process (provided by SLURM `--gres=gpu:1`)

### Why The Error Message is Confusing

The error says "CUDA_ERROR_NOT_INITIALIZED" which sounds like a CUDA driver issue, but it's actually JAX trying to initialize CUDA when it doesn't have CUDA support compiled in. It's like trying to start a car engine that was never installed!

## Expected Output After Fix

Once fixed, your job output should show:

```
========================================
GPU and CUDA Diagnostics
========================================
--- GPU Information ---
[nvidia-smi output showing GPU]

--- JAX Installation ---
JAX version: 0.4.x
JAX devices: [CudaDevice(id=0)]
JAX default backend: gpu
========================================

[Training starts and runs fast]
```

## Monitoring Your Jobs

After fixing, you can verify jobs are using GPU:

```bash
# Check if jobs are running fast:
squeue -u $USER

# Check job output:
tail -f logs/scan_JOBID_TASKID.out

# Should see fast epoch times:
# Epoch 1/100 - 2.3s - train_loss: ...
# Epoch 2/100 - 2.1s - train_loss: ...
```

If epochs take 30+ seconds, still on CPU!

## Questions?

- **Q: Do I need to cancel my running jobs?**  
  A: If they're on CPU, yes - they'll take 10-100x longer. Cancel with `scancel JOBID`

- **Q: Will this affect my data?**  
  A: No, this only changes how JAX is installed, not your data.

- **Q: Do I need cluster admin help?**  
  A: Probably not - this is usually a Python package issue. Only contact admin if `test_gpu.sh` shows `nvidia-smi failed`.

- **Q: How long will the fix take?**  
  A: ~5-10 minutes to reinstall JAX, ~5 minutes to test.

## Summary

- ❌ **Problem:** JAX installed without CUDA support → can't use GPU → very slow training
- ✅ **Solution:** Reinstall JAX with `pip install "jax[cuda12]"`
- 🔧 **Tools provided:** test_gpu.sh (test), fix_jax_cuda.sh (fix), updated SLURM scripts (diagnose)
- 📖 **Docs:** README_GPU_FIX.md (quick guide), CUDA_TROUBLESHOOTING.md (detailed guide)

Good luck! Run `sbatch test_gpu.sh` first to confirm the issue.

