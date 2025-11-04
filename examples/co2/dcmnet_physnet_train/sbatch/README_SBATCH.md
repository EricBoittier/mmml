# SBATCH Scripts for Scicore HPC

Foolproof scripts for running your DCMNet/PhysNet training on Scicore HPC cluster.

## 📋 Quick Start

### 1. Setup (First Time Only)

```bash
# Navigate to the sbatch directory
cd /path/to/mmml/examples/co2/dcmnet_physnet_train/sbatch

# Create logs directory
mkdir -p logs

# Test your environment
sbatch 05_gpu_test.sbatch
```

Check the test output:
```bash
tail -f logs/test_gpu_*.out
```

### 2. Edit Data Paths

Before running training, edit the data file paths in each script:

```bash
# Edit these variables in each .sbatch file:
TRAIN_EFD="train.npz"          # Change to your path
TRAIN_ESP="grids_train.npz"    # Change to your path
VALID_EFD="valid.npz"          # Change to your path
VALID_ESP="grids_valid.npz"    # Change to your path
```

**Pro tip:** Use absolute paths on Scicore, e.g.:
```bash
TRAIN_EFD="/scicore/home/yourgroup/yourusername/data/train.npz"
```

### 3. Run Training

```bash
# Quick test (4 hours, 50 epochs)
sbatch 01_train_dcmnet_quick.sbatch

# Full training (24 hours, 200 epochs)
sbatch 02_train_dcmnet_full.sbatch

# Non-equivariant model
sbatch 03_train_noneq_model.sbatch

# Model comparison
sbatch 04_compare_models.sbatch
```

---

## 📄 Script Descriptions

### `01_train_dcmnet_quick.sbatch`
- **Purpose:** Quick training run for testing/debugging
- **Time:** 4 hours
- **Epochs:** 50
- **QoS:** gpu6hours
- **Use for:** Testing setup, quick experiments

### `02_train_dcmnet_full.sbatch`
- **Purpose:** Full production training
- **Time:** 24 hours
- **Epochs:** 200
- **QoS:** gpu24hours
- **Use for:** Final models, publication results

### `03_train_noneq_model.sbatch`
- **Purpose:** Train non-equivariant baseline model
- **Time:** 12 hours
- **Epochs:** 150
- **QoS:** gpu12hours
- **Use for:** Comparison baseline

### `04_compare_models.sbatch`
- **Purpose:** Head-to-head model comparison with equivariance tests
- **Time:** 12 hours
- **Epochs:** 100 (each model)
- **QoS:** gpu12hours
- **Use for:** Research comparisons, model selection

### `05_gpu_test.sbatch`
- **Purpose:** Quick environment and GPU test
- **Time:** 10 minutes
- **Use for:** Verifying setup before long runs

### `06_hyperparameter_array.sbatch`
- **Purpose:** Parallel hyperparameter sweep
- **Time:** 8 hours per configuration
- **Array:** 9 jobs (3 learning rates × 3 batch sizes)
- **Use for:** Finding optimal hyperparameters

---

## 🔧 Configuration

### Module Loading

Each script includes:
```bash
module purge
module load Python/3.11
source activate mmml
```

**Adjust if needed:**
- If you use different Python version: change `Python/3.11`
- If you need CUDA modules: add `module load CUDA/12.0` or similar
- If your conda env has different name: change `mmml`

### Resource Requests

Default resources (can be adjusted in each script):

| Resource | Quick | Full | Compare |
|----------|-------|------|---------|
| Time | 4h | 24h | 12h |
| Memory | 32GB | 64GB | 64GB |
| CPUs | 4 | 8 | 8 |
| GPUs | 1 | 1 | 1 |
| QoS | gpu6hours | gpu24hours | gpu12hours |

**To change resources**, edit the `#SBATCH` directives:
```bash
#SBATCH --mem=128G         # For more memory
#SBATCH --cpus-per-task=16 # For more CPUs
#SBATCH --time=48:00:00    # For longer time
```

### Training Parameters

Each script has a configuration section:
```bash
# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
TRAIN_EFD="train.npz"
TRAIN_ESP="grids_train.npz"
VALID_EFD="valid.npz"
VALID_ESP="grids_valid.npz"

# Training parameters
EPOCHS=200
BATCH_SIZE=8
OPTIMIZER="adamw"
```

Edit these values to customize your run.

---

## 📊 Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View running job output
```bash
# While job is running
tail -f logs/train_dcmnet_full_JOBID.out

# After job completes
less logs/train_dcmnet_full_JOBID.out
```

### Cancel a job
```bash
scancel JOBID
```

### View job details
```bash
scontrol show job JOBID
```

---

## 📁 Output Locations

### Logs
- Standard output: `logs/scriptname_JOBID.out`
- Error output: `logs/scriptname_JOBID.err`

### Checkpoints
- Training checkpoints: `../checkpoints/MODEL_NAME/`
- Best model: `../checkpoints/MODEL_NAME/best_params.pkl`

### Comparison Results
- Comparison outputs: `../comparisons/COMPARISON_NAME/`
- Plots: `../comparisons/COMPARISON_NAME/*.png`
- Metrics: `../comparisons/COMPARISON_NAME/comparison_results.json`

---

## 🔄 Common Workflows

### Workflow 1: First-Time Training

```bash
# 1. Test environment
sbatch 05_gpu_test.sbatch

# 2. Quick training test
sbatch 01_train_dcmnet_quick.sbatch

# 3. Full training (if quick test works)
sbatch 02_train_dcmnet_full.sbatch
```

### Workflow 2: Model Comparison

```bash
# Run comparison (trains both models)
sbatch 04_compare_models.sbatch

# Check results
ls -lh comparisons/model_comparison_*/
```

### Workflow 3: Hyperparameter Tuning

```bash
# Submit job array
sbatch 06_hyperparameter_array.sbatch

# Monitor progress
watch -n 60 squeue -u $USER

# After completion, compare checkpoints
ls -lh checkpoints/hparam_*/
```

### Workflow 4: Resume Training

To resume from a checkpoint, add to your sbatch script:
```bash
python trainer.py \
    ... \
    --restart-from checkpoints/previous_run/best_params.pkl \
    --start-epoch 101
```

---

## 🐛 Troubleshooting

### Problem: Module not found errors

**Solution:** Check module names for your Scicore setup:
```bash
module avail Python
module avail CUDA
```

Update script accordingly:
```bash
module load Python/3.11.5  # Use exact version
```

### Problem: Out of memory errors

**Solution:** Reduce batch size or request more memory:
```bash
#SBATCH --mem=128G
# And/or in the script:
BATCH_SIZE=2
```

### Problem: JAX not finding GPU

**Solution:** Verify CUDA modules:
```bash
# Add to script before training:
module load CUDA/12.0
module load cuDNN/8.9.0

# Verify in Python:
python -c "import jax; print(jax.devices())"
```

### Problem: Data files not found

**Solution:** Use absolute paths:
```bash
TRAIN_EFD="/scicore/home/mygroup/myuser/data/train.npz"
```

Or navigate to data directory:
```bash
cd /path/to/data
sbatch /path/to/sbatch/script.sbatch
```

### Problem: Permission denied on logs

**Solution:** Create logs directory with correct permissions:
```bash
mkdir -p logs
chmod 755 logs
```

---

## 💡 Tips & Best Practices

### 1. Always Test First
```bash
sbatch 05_gpu_test.sbatch  # Before any long run
```

### 2. Use Job Names
Makes monitoring easier:
```bash
#SBATCH --job-name=myproject_dcmnet
```

### 3. Email Notifications (Optional)
Add to sbatch script:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com
```

### 4. Checkpoint Frequently
Default is every epoch. For very long runs, ensure checkpointing works:
```bash
# Check checkpoint directory exists
ls -lh checkpoints/
```

### 5. Monitor GPU Usage
Add to script:
```bash
nvidia-smi
```

Or monitor during run:
```bash
ssh NODE_NAME
nvidia-smi -l 5
```

### 6. Use Job Dependencies
Run jobs in sequence:
```bash
# Submit first job
JOBID1=$(sbatch --parsable 01_train_dcmnet_quick.sbatch)

# Submit second job that waits for first
sbatch --dependency=afterok:$JOBID1 02_train_dcmnet_full.sbatch
```

---

## 🎯 Partition & QoS Selection Guide

| Task | Time Needed | Partition | QoS |
|------|-------------|-----------|-----|
| Quick test | < 1 hour | titan | gpu6hours |
| Development | 1-6 hours | titan | gpu6hours |
| Standard training | 6-12 hours | titan | gpu12hours |
| Full training | 12-24 hours | titan | gpu24hours |
| Long training | > 24 hours | titan | gpu48hours* |

*Check Scicore documentation for available QoS levels

---

## 📞 Getting Help

1. **Scicore Documentation:** https://wiki.scicore.unibas.ch/
2. **Check logs:** Always check both `.out` and `.err` files
3. **Test scripts:** Use `05_gpu_test.sbatch` to verify environment
4. **Project docs:** See `../README.md`, `../QUICK_REFERENCE.md`

---

## ✅ Checklist Before First Run

- [ ] Created `logs/` directory
- [ ] Edited data paths in sbatch scripts
- [ ] Verified conda environment name (`mmml`)
- [ ] Tested environment (`05_gpu_test.sbatch`)
- [ ] Checked data files exist
- [ ] Reviewed resource requests (memory, time, CPUs)
- [ ] Set appropriate job name
- [ ] Verified partition and QoS are correct

---

## 🚀 You're Ready!

Start with:
```bash
sbatch 05_gpu_test.sbatch
```

Then proceed to training once the test passes! 🎉

