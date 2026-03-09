# 🚀 Quick Start - Scicore HPC

## 1️⃣ First Time Setup (5 minutes)

```bash
# On Scicore login node
cd /path/to/mmml/examples/co2/dcmnet_physnet_train/sbatch

# Load conda
module load Anaconda3  # or Miniconda3

# Create environment (interactive)
bash setup_environment.sh

# Create logs directory
mkdir -p logs
```

## 2️⃣ Test Everything (10 minutes)

```bash
# Submit test job
sbatch 05_gpu_test.sbatch

# Wait ~2 minutes, then check output
tail -f logs/test_gpu_*.out

# Should see ✅ for all tests
```

## 3️⃣ Edit Data Paths (2 minutes)

Open any training script and update these lines:

```bash
TRAIN_EFD="/full/path/to/train.npz"
TRAIN_ESP="/full/path/to/grids_train.npz"
VALID_EFD="/full/path/to/valid.npz"
VALID_ESP="/full/path/to/grids_valid.npz"
```

💡 **Tip:** Use absolute paths on Scicore!

## 4️⃣ Run Training

```bash
# Quick test (4 hours)
sbatch 01_train_dcmnet_quick.sbatch

# Monitor progress
tail -f logs/train_dcmnet_quick_*.out

# Check job status
squeue -u $USER
```

---

## 📝 All Available Scripts

| Script | Time | Purpose | When to Use |
|--------|------|---------|-------------|
| `05_gpu_test.sbatch` | 10min | Test setup | **Always run this first** |
| `01_train_dcmnet_quick.sbatch` | 4h | Quick training | Testing, debugging |
| `02_train_dcmnet_full.sbatch` | 24h | Full training | Production models |
| `03_train_noneq_model.sbatch` | 12h | Non-equivariant | Baseline comparison |
| `04_compare_models.sbatch` | 12h | Model comparison | Research, evaluation |
| `06_hyperparameter_array.sbatch` | 8h × 9 | Hyperparameter sweep | Optimization |

---

## 🎯 Common Commands

```bash
# Submit job
sbatch SCRIPT.sbatch

# Check status
squeue -u $USER

# View output
tail -f logs/JOBNAME_JOBID.out

# Cancel job
scancel JOBID

# Check results
ls -lh checkpoints/
ls -lh comparisons/
```

---

## ⚡ Pro Tips

1. **Always test first:** `sbatch 05_gpu_test.sbatch`
2. **Use absolute paths** for data files
3. **Monitor your job:** `tail -f logs/*.out`
4. **Check checkpoints:** Models save to `checkpoints/MODEL_NAME/`
5. **Email notifications:** Add `#SBATCH --mail-user=your@email.com` to script

---

## 🆘 Common Issues

### "Command not found: conda"
```bash
module load Anaconda3
```

### "Data file not found"
→ Use absolute paths in sbatch scripts

### "Out of memory"
→ Edit script: `#SBATCH --mem=128G` and/or reduce `BATCH_SIZE=2`

### "JAX not finding GPU"
→ Check test output: `sbatch 05_gpu_test.sbatch`

---

## 📚 Full Documentation

See `README_SBATCH.md` for complete documentation including:
- Detailed script descriptions
- Resource configuration
- Troubleshooting guide
- Advanced workflows
- Tips & best practices

---

## ✅ Before Your First Run

- [ ] Environment created (`setup_environment.sh`)
- [ ] GPU test passed (`05_gpu_test.sbatch`)
- [ ] Data paths edited in scripts
- [ ] `logs/` directory created
- [ ] Data files verified to exist

---

**Ready?** Start here:

```bash
sbatch 05_gpu_test.sbatch
```

🎉 **You're all set!**

