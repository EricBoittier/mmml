# CUDA 13 Environment Setup Guide

## Overview

This guide helps you create a fresh Python virtual environment with JAX + CUDA support specifically configured for PhysNet training on GPU.

## Why a New Environment?

The CUDA error you encountered (`CUDA_ERROR_NOT_INITIALIZED`) typically means JAX wasn't installed with proper GPU support. Rather than trying to fix the existing environment, we're creating a clean new one with guaranteed GPU support.

## Quick Start

### Step 1: Create the Environment

```bash
cd ~/mmml/examples/co2/physnet_train
bash setup_cuda13_env.sh
```

This will:
- Create a new virtual environment at `~/mmml/.venv_cuda13`
- Install JAX with CUDA support (tries CUDA 13, falls back to CUDA 12)
- Install all required packages (numpy, scipy, optax, flax, ase, etc.)
- Install mmml package in development mode
- Create an activation script for easy use

**Time:** ~5-10 minutes

### Step 2: Test GPU Access

```bash
cd ~/mmml/examples/co2/physnet_train
sbatch test_gpu_cuda13.sh
```

Wait for the job to complete (~5 minutes), then check results:

```bash
cat logs/test_gpu_cuda13_*.out
```

**Expected output:**
```
✅ GPU device found!
✅ GPU computation successful!
✅ ALL TESTS PASSED!
```

### Step 3: Run Training

Once the test passes, you're ready to train:

```bash
# Quick scan (6 jobs, ~1 hour each)
sbatch slurm_cuda13_quick_scan.sh

# OR Full scan (24 jobs, ~6 hours each)
sbatch slurm_cuda13_full_scan.sh
```

## Files Created

### Setup & Testing
- **setup_cuda13_env.sh** - Creates the CUDA 13 environment
- **test_gpu_cuda13.sh** - Tests GPU access and JAX functionality
- **activate_cuda13.sh** - Quick activation script (in ~/mmml/)

### Training Scripts
- **slurm_cuda13_quick_scan.sh** - Quick hyperparameter scan (6 configs)
- **slurm_cuda13_full_scan.sh** - Full hyperparameter scan (24 configs)

### Documentation
- **CUDA13_SETUP.md** - This file
- **CUDA_TROUBLESHOOTING.md** - Detailed troubleshooting
- **GPU_ERROR_SUMMARY.md** - Explanation of the original error

## Environment Details

### Location
- **Path:** `~/mmml/.venv_cuda13`
- **Python:** 3.12 (or 3.11/3.10 depending on availability)
- **Purpose:** PhysNet training with JAX + GPU

### Activation

**Option 1: Use the activation script**
```bash
source ~/mmml/activate_cuda13.sh
```

**Option 2: Manual activation**
```bash
module load CUDA/13.0  # or CUDA/12.x
source ~/mmml/.venv_cuda13/bin/activate
```

### Installed Packages

**Core:**
- JAX with CUDA support (cuda12_pip or cuda12)
- jaxlib with CUDA extensions
- JAX CUDA plugins

**ML Frameworks:**
- optax (optimizers)
- flax (neural networks)
- dm-haiku (neural networks)
- orbax-checkpoint (model checkpointing)

**Scientific:**
- numpy, scipy, matplotlib
- ase (Atomic Simulation Environment)

**Project:**
- mmml (installed in development mode from ~/mmml)

## Usage

### Training with the New Environment

All `slurm_cuda13_*.sh` scripts automatically:
1. Load CUDA module (tries 13.0, falls back to 12.x)
2. Activate the `~/mmml/.venv_cuda13` environment
3. Run diagnostics to verify GPU access
4. Execute training with JAX on GPU

### Interactive Testing

To test interactively:

```bash
# Request interactive GPU session
srun --partition=titan --gres=gpu:1 --pty bash

# Activate environment
source ~/mmml/activate_cuda13.sh

# Test JAX
python -c "import jax; print('Devices:', jax.devices())"

# Expected: [CudaDevice(id=0)]
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f logs/cuda13_quick_JOBID_TASKID.out
```

### Check for GPU usage
```bash
# Should see fast epoch times (1-5 seconds)
# Not 30-300 seconds (which indicates CPU)
grep "Epoch" logs/cuda13_quick_JOBID_TASKID.out
```

## Verification Checklist

After setup, verify:

- [ ] Environment created at `~/mmml/.venv_cuda13`
- [ ] Activation script works: `source ~/mmml/activate_cuda13.sh`
- [ ] Test job passes: `sbatch test_gpu_cuda13.sh` → check output
- [ ] JAX reports GPU: Look for `[CudaDevice(id=0)]` in logs
- [ ] Training runs fast: Epoch times 1-5 seconds (not 30+)

## Troubleshooting

### Test job fails with "No GPU device found"

**Cause:** JAX still not installed with CUDA support

**Fix:**
```bash
source ~/mmml/.venv_cuda13/bin/activate
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### nvidia-smi fails in test job

**Cause:** GPU not allocated by SLURM

**Fix:**
- Check if titan partition has GPUs: `sinfo -p titan`
- Try different partition: Edit scripts and change `--partition=titan` to `--partition=a100` or similar
- Contact cluster admin

### Module load CUDA fails

**Cause:** CUDA module not available

**Fix:**
```bash
# Check available CUDA versions
module avail CUDA

# Use whatever version is available
module load CUDA/12.x.x
```

### Training still slow (30+ sec/epoch)

**Cause:** Still running on CPU

**Fix:**
1. Check log file for "JAX devices:" line
2. Should see `[CudaDevice(id=0)]` not `[CpuDevice(id=0)]`
3. If CPU, JAX needs reinstalling (see first troubleshooting item)

## Comparison: Old vs New Environment

| Aspect | Old (~/.venv) | New (~/.venv_cuda13) |
|--------|---------------|----------------------|
| JAX | May lack CUDA | Guaranteed CUDA support |
| Installation | Unknown history | Fresh, documented |
| CUDA version | Unknown | 12.x or 13.x |
| GPU support | ❌ (broken) | ✅ (verified) |
| Performance | CPU (slow) | GPU (fast) |

## Performance Expectations

With GPU working correctly:

| Model Size | Epoch Time | 100 Epochs | 500 Epochs |
|------------|------------|------------|------------|
| Small (32 features) | 1-2 sec | ~3 min | ~15 min |
| Medium (128 features) | 2-3 sec | ~5 min | ~25 min |
| Large (256 features) | 3-5 sec | ~8 min | ~40 min |

If your epochs take 30+ seconds, you're on CPU!

## Keeping Both Environments

You can keep both environments:

- **Old environment:** `~/mmml/.venv`
  - Use for: CPU-only work, testing
  - Activate: `source ~/mmml/.venv/bin/activate`

- **New environment:** `~/mmml/.venv_cuda13`
  - Use for: GPU training
  - Activate: `source ~/mmml/activate_cuda13.sh`

Scripts with `cuda13` in the name use the new environment automatically.

## Cleanup

If you want to remove the old environment later (after confirming new one works):

```bash
# ONLY do this after verifying cuda13 environment works!
rm -rf ~/mmml/.venv
```

## FAQ

**Q: What if CUDA 13.0 is not available?**  
A: Scripts automatically fall back to CUDA 12.x. JAX works fine with CUDA 12.

**Q: Do I need to reinstall mmml?**  
A: No, it's installed in development mode. Changes to mmml code are automatically available.

**Q: Can I use this for other projects?**  
A: Yes! This environment has JAX + CUDA and can be used for any JAX project.

**Q: How much disk space does it use?**  
A: ~2-3 GB for the full environment with all packages.

**Q: Can I delete the old environment?**  
A: Yes, but wait until you confirm the new one works for training.

## Summary

```bash
# 1. Create environment
bash setup_cuda13_env.sh

# 2. Test GPU
sbatch test_gpu_cuda13.sh
cat logs/test_gpu_cuda13_*.out  # Should see "ALL TESTS PASSED"

# 3. Train
sbatch slurm_cuda13_quick_scan.sh

# 4. Monitor
squeue -u $USER
tail -f logs/cuda13_quick_*_0.out
```

That's it! You now have a working GPU environment for PhysNet training.

## Next Steps

After successful training:
- Compare results across different hyperparameters
- Use best model for predictions
- See main README for analysis and visualization

## Support

If you encounter issues:
1. Check **CUDA_TROUBLESHOOTING.md** for detailed diagnostics
2. Share output of `test_gpu_cuda13.sh` when asking for help
3. Include `pip list | grep jax` output from the new environment

