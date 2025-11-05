# Complete Guide: Running on Scicore HPC

**Everything you need to run your code on Scicore HPC cluster.**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start) (5 minutes)
2. [File Structure](#file-structure)
3. [Workflow Overview](#workflow-overview)
4. [Detailed Steps](#detailed-steps)
5. [Plotting Results](#plotting-results)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Setup (first time only)
cd sbatch
bash setup_environment.sh
mkdir -p logs

# 2. Test
sbatch 05_gpu_test.sbatch

# 3. Edit data paths in sbatch scripts
# Open any .sbatch file and update these:
TRAIN_EFD="/path/to/train.npz"
TRAIN_ESP="/path/to/grids_train.npz"
VALID_EFD="/path/to/valid.npz"
VALID_ESP="/path/to/grids_valid.npz"

# 4. Run
sbatch 01_train_dcmnet_quick.sbatch

# 5. Plot results
python plot_comparison_results.py comparisons/*/comparison_results.json
```

---

## 📂 File Structure

```
dcmnet_physnet_train/
│
├── sbatch/                                    # HPC submission scripts
│   ├── 01_train_dcmnet_quick.sbatch          # Quick training (4h)
│   ├── 02_train_dcmnet_full.sbatch           # Full training (24h)
│   ├── 03_train_noneq_model.sbatch           # Non-equivariant model
│   ├── 04_compare_models.sbatch              # Model comparison
│   ├── 05_gpu_test.sbatch                    # Environment test
│   ├── 06_hyperparameter_array.sbatch        # Hyperparameter sweep
│   ├── setup_environment.sh                   # Conda setup script
│   ├── README_SBATCH.md                       # Complete sbatch docs
│   └── QUICK_START.md                         # Quick reference
│
├── plot_comparison_results.py                 # Plotting CLI tool
├── PLOTTING_CLI_GUIDE.md                      # Complete plotting docs
├── PLOTTING_CLI_EXAMPLES.md                   # Quick plotting examples
│
├── trainer.py                                 # Main training script
├── compare_models.py                          # Model comparison script
│
├── logs/                                      # Job output logs
│   ├── train_dcmnet_quick_JOBID.out
│   ├── train_dcmnet_quick_JOBID.err
│   └── ...
│
├── checkpoints/                               # Model checkpoints
│   └── MODEL_NAME/
│       ├── best_params.pkl
│       └── ...
│
└── comparisons/                               # Comparison results
    └── COMPARISON_NAME/
        ├── comparison_results.json
        ├── performance_comparison.png
        ├── efficiency_comparison.png
        ├── equivariance_comparison.png
        └── overview_combined.png
```

---

## 🔄 Workflow Overview

### Simple Training Workflow

```
1. Setup Environment          → bash setup_environment.sh
2. Test GPU/Environment       → sbatch 05_gpu_test.sbatch
3. Quick Training Test        → sbatch 01_train_dcmnet_quick.sbatch
4. Full Training              → sbatch 02_train_dcmnet_full.sbatch
5. Plot/Analyze Results       → python plot_comparison_results.py results.json
```

### Model Comparison Workflow

```
1. Run Comparison             → sbatch 04_compare_models.sbatch
2. Wait for Completion        → squeue -u $USER
3. Check Summary              → python plot_comparison_results.py results.json --summary-only
4. Generate Plots             → python plot_comparison_results.py results.json
5. Analyze Results            → View plots in comparisons/
```

### Hyperparameter Tuning Workflow

```
1. Submit Array Job           → sbatch 06_hyperparameter_array.sbatch
2. Monitor Progress           → watch -n 60 squeue -u $USER
3. Wait for All Jobs          → (all 9 jobs complete)
4. Compare Results            → python plot_comparison_results.py \
                                  comparisons/*/comparison_results.json \
                                  --compare-multiple
5. Select Best Model          → Based on plots
```

---

## 📖 Detailed Steps

### Step 1: Initial Setup (First Time Only)

#### On Scicore Login Node:

```bash
# Navigate to your project
cd /path/to/mmml/examples/co2/dcmnet_physnet_train

# Load conda
module load Anaconda3  # or Miniconda3

# Create conda environment
cd sbatch
bash setup_environment.sh
```

**This will:**
- Create `mmml` conda environment
- Install all dependencies
- Verify installation

**Expected time:** 10-15 minutes

---

### Step 2: Test Your Setup

```bash
# Create logs directory
mkdir -p logs

# Submit test job
sbatch 05_gpu_test.sbatch

# Wait ~2 minutes, then check
tail -f logs/test_gpu_*.out
```

**Look for:**
- ✅ JAX computation test passed
- ✅ GPU devices found
- ✅ All packages imported

**If test fails:** See [Troubleshooting](#troubleshooting)

---

### Step 3: Configure Data Paths

Edit each `.sbatch` file you plan to use:

```bash
# Open in your editor
nano 01_train_dcmnet_quick.sbatch

# Find and update these lines:
TRAIN_EFD="/full/path/to/train.npz"
TRAIN_ESP="/full/path/to/grids_train.npz"
VALID_EFD="/full/path/to/valid.npz"
VALID_ESP="/full/path/to/grids_valid.npz"
```

**Important:** Use absolute paths on Scicore!

```bash
# Good
TRAIN_EFD="/scicore/home/mygroup/myuser/data/train.npz"

# Bad
TRAIN_EFD="train.npz"
TRAIN_EFD="../data/train.npz"
```

---

### Step 4: Submit Training Job

#### Quick Test (4 hours, 50 epochs):

```bash
sbatch 01_train_dcmnet_quick.sbatch
```

#### Full Training (24 hours, 200 epochs):

```bash
sbatch 02_train_dcmnet_full.sbatch
```

#### Model Comparison:

```bash
sbatch 04_compare_models.sbatch
```

---

### Step 5: Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/train_dcmnet_quick_JOBID.out

# Check progress
grep "Epoch" logs/train_dcmnet_quick_JOBID.out | tail -5
```

---

### Step 6: Analyze Results

#### Quick Summary:

```bash
python plot_comparison_results.py \
    comparisons/my_comparison/comparison_results.json \
    --summary-only
```

#### Generate All Plots:

```bash
python plot_comparison_results.py \
    comparisons/my_comparison/comparison_results.json
```

#### High-Resolution for Papers:

```bash
python plot_comparison_results.py \
    comparisons/my_comparison/comparison_results.json \
    --format pdf --dpi 300 \
    --output-dir publication_figures
```

---

## 📊 Plotting Results

### Basic Usage

```bash
# Plot everything
python plot_comparison_results.py comparison_results.json

# Just text summary
python plot_comparison_results.py comparison_results.json --summary-only

# Specific plot type
python plot_comparison_results.py comparison_results.json --plot-type performance
```

### Advanced Options

```bash
# High-res PDF
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --output-dir paper_figs

# Custom colors
python plot_comparison_results.py results.json \
    --colors "#FF6B6B,#4ECDC4"

# Compare multiple runs
python plot_comparison_results.py \
    run1/results.json run2/results.json run3/results.json \
    --compare-multiple --metric dipole_mae
```

**See `PLOTTING_CLI_GUIDE.md` for complete documentation.**

---

## 🎯 Common Scenarios

### Scenario 1: First Time User

```bash
# 1. Setup
cd sbatch && bash setup_environment.sh

# 2. Test
sbatch 05_gpu_test.sbatch

# 3. Edit data paths in 01_train_dcmnet_quick.sbatch

# 4. Quick training
sbatch 01_train_dcmnet_quick.sbatch

# 5. Check results
tail -f logs/train_dcmnet_quick_*.out
```

### Scenario 2: Production Training

```bash
# 1. Edit 02_train_dcmnet_full.sbatch with your data paths

# 2. Submit job
sbatch 02_train_dcmnet_full.sbatch

# 3. Monitor
squeue -u $USER

# 4. After completion, check checkpoint
ls -lh checkpoints/dcmnet_full_*/
```

### Scenario 3: Model Comparison for Paper

```bash
# 1. Run comparison
sbatch 04_compare_models.sbatch

# 2. Wait for completion (~12 hours)

# 3. Generate publication figures
python plot_comparison_results.py \
    comparisons/model_comparison_*/comparison_results.json \
    --format pdf --dpi 300 --output-dir paper_figures

# 4. Download figures
# On local machine:
# scp -r scicore:/path/to/paper_figures ./
```

### Scenario 4: Hyperparameter Search

```bash
# 1. Edit 06_hyperparameter_array.sbatch
#    Customize learning rates and batch sizes

# 2. Submit array job (9 parallel jobs)
sbatch 06_hyperparameter_array.sbatch

# 3. Monitor all jobs
watch -n 60 "squeue -u $USER"

# 4. After all complete, compare
python plot_comparison_results.py \
    comparisons/hparam_*/comparison_results.json \
    --compare-multiple --metric dipole_mae \
    --output-dir hparam_analysis

# 5. Review and select best
ls -lh hparam_analysis/
```

---

## 🐛 Troubleshooting

### Problem: GPU test fails

**Check:**
```bash
# View error log
cat logs/test_gpu_*.err

# Check GPU partition
sinfo -p titan

# Verify modules
module list
```

**Solution:**
```bash
# Reload modules
module purge
module load Python/3.11
module load CUDA/12.0  # if needed
```

### Problem: Data files not found

**Error:**
```
❌ Error: Train EFD file not found: train.npz
```

**Solution:**
1. Use absolute paths
2. Verify files exist:
   ```bash
   ls -lh /path/to/train.npz
   ```
3. Update paths in sbatch script

### Problem: Out of memory

**Error:**
```
slurmstepd: error: Exceeded job memory limit
```

**Solution:**
Edit sbatch script:
```bash
#SBATCH --mem=128G    # Increase from 64G
```
And/or reduce batch size:
```bash
BATCH_SIZE=2          # Reduce from 4 or 8
```

### Problem: Job exceeds time limit

**Solution:**
Request more time:
```bash
#SBATCH --time=48:00:00    # Instead of 24:00:00
#SBATCH --qos=gpu48hours   # If available
```

Or reduce epochs:
```bash
EPOCHS=100                  # Instead of 200
```

### Problem: Environment not found

**Error:**
```
conda: command not found
```

**Solution:**
```bash
module load Anaconda3
source activate mmml
```

### Problem: Plots look bad

**Solutions:**
```bash
# Increase DPI
--dpi 300

# Increase figure size
--figsize 16,12

# Try different colors
--colors "#FF6B6B,#4ECDC4"
```

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| `SCICORE_COMPLETE_GUIDE.md` | **This file** - Complete workflow |
| `sbatch/README_SBATCH.md` | Complete sbatch script documentation |
| `sbatch/QUICK_START.md` | Quick reference for sbatch scripts |
| `PLOTTING_CLI_GUIDE.md` | Complete plotting tool documentation |
| `PLOTTING_CLI_EXAMPLES.md` | Quick plotting examples |
| `COMPARISON_GUIDE.md` | Model comparison details |
| `MODEL_OPTIONS.md` | Model architecture options |
| `OPTIMIZER_GUIDE.md` | Optimizer selection guide |
| `QUICK_REFERENCE.md` | One-page cheat sheet |

---

## ✅ Checklist

### Before First Run:
- [ ] Environment created (`bash setup_environment.sh`)
- [ ] GPU test passed (`sbatch 05_gpu_test.sbatch`)
- [ ] Data files verified to exist
- [ ] Data paths updated in sbatch scripts
- [ ] `logs/` directory created
- [ ] Reviewed resource requests (memory, time)

### Before Production Run:
- [ ] Quick test successful
- [ ] Checkpoints saving correctly
- [ ] Output looks reasonable
- [ ] Sufficient time allocated
- [ ] Sufficient memory allocated

### After Run Completes:
- [ ] Check exit code (should be 0)
- [ ] Verify checkpoint exists
- [ ] Generate plots
- [ ] Save results

---

## 🎓 Learning Path

1. **Beginner:** Start with `sbatch/QUICK_START.md`
2. **Intermediate:** Read `COMPARISON_GUIDE.md` and `PLOTTING_CLI_EXAMPLES.md`
3. **Advanced:** Explore `MODEL_OPTIONS.md` and `OPTIMIZER_GUIDE.md`
4. **Expert:** Customize sbatch scripts and create your own workflows

---

## 💡 Tips for Success

1. **Always test first** - Use `05_gpu_test.sbatch` before long runs
2. **Use absolute paths** - Avoid relative paths on HPC
3. **Monitor jobs** - Check logs regularly
4. **Save checkpoints** - Verify they're being created
5. **Start small** - Quick test before full training
6. **Document runs** - Keep notes on what works
7. **Backup results** - Download important checkpoints/plots
8. **Check quotas** - Monitor disk space usage

---

## 🚀 Next Steps

After successful training:

1. **Evaluate model:**
   - Use evaluation scripts in project
   - Run dynamics simulations
   - Calculate spectroscopic properties

2. **Refine model:**
   - Try different hyperparameters
   - Adjust architecture
   - Add more training data

3. **Publish results:**
   - Generate high-res figures
   - Write up findings
   - Share code/models

---

## 📞 Getting Help

1. **Check logs:** Always look at `.out` and `.err` files first
2. **Test script:** Run `05_gpu_test.sbatch` to verify environment
3. **Documentation:** See index above for relevant docs
4. **Scicore wiki:** https://wiki.scicore.unibas.ch/

---

**You're all set! Start with the Quick Start section above. Good luck! 🎉**

