# ✅ Glycol Dataset Ready for Training!

Your dataset has been cleaned and validated. You're ready to start training!

## Quick Status

- ✅ **Dataset cleaned:** 5,895 structures (99.85% retained)
- ✅ **Fields validated:** E, F, R, Z, N, D, Dxyz
- ✅ **SCF failures removed:** 9 bad structures eliminated
- ✅ **Auto-detection:** `num_atoms` will be automatically detected (60)
- ✅ **File size:** 2.0 MB (compressed, efficient)

## Start Training Now

Simply run this command - no `--num_atoms` needed:

```bash
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_run1 \
  --n_train 4000 \
  --n_valid 800 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --features 128 \
  --num_iterations 3 \
  --cutoff 10.0 \
  --ckpt_dir checkpoints/glycol_run1
```

**What you'll see:**
```
Auto-detecting number of atoms from dataset...
  ✅ Detected num_atoms = 60 from R.shape (includes padding)
     (Note: Actual molecule size = 10, padded to 60)
```

## What Was Fixed

### Issue 1: Manual num_atoms Specification ❌→✅
**Before:** Had to manually figure out and specify `--num_atoms 60`  
**Fix:** Training script now auto-detects from R.shape[1]  

### Issue 2: Extra Fields Crash ❌→✅
**Before:** Training crashed with `IndexError: index 1505 is out of bounds`  
**Fix:** Cleaning script keeps only essential fields (E, F, R, Z, N, D, Dxyz)  

### Issue 3: Orbax Path Error ❌→✅
**Before:** `ValueError: Checkpoint path should be absolute`  
**Fix:** Training script converts paths to absolute automatically  

### Issue 4: Over-Aggressive Cleaning ❌→✅
**Before:** Removed 30% of data (1,774 structures)  
**Fix:** Use `--no-check-distances` to keep 99.85% (only remove SCF failures)  

## Monitoring Training

While training runs, monitor progress:

```bash
# In another terminal
watch -n 10 'python -m mmml.cli.plot_training \
  checkpoints/glycol_run1/history.json \
  --output-dir plots --quiet'
```

## After Training

### Test the Model
```bash
python -m mmml.cli.calculator \
  --checkpoint checkpoints/glycol_run1 \
  --test-molecule H2O
```

### Analyze Dynamics
```bash
python -m mmml.cli.dynamics \
  --checkpoint checkpoints/glycol_run1 \
  --molecule CO2 \
  --optimize --frequencies --ir-spectra \
  --output-dir analysis
```

## Training Tips

**Start small for testing:**
```bash
# 5-minute test run
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag test_quick \
  --n_train 100 \
  --n_valid 20 \
  --num_epochs 5 \
  --batch_size 4 \
  --ckpt_dir /tmp/test_glycol
```

**Full production run:**
```bash
# Use most of your data
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_production \
  --n_train 4700 \
  --n_valid 1000 \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --features 128 \
  --num_iterations 3 \
  --cutoff 10.0 \
  --ckpt_dir checkpoints/glycol_production
```

## Expected Training Time

- **Test run** (100 samples, 5 epochs): ~5 minutes
- **Medium run** (3000 samples, 50 epochs): ~2-4 hours (GPU)
- **Full run** (4700 samples, 100 epochs): ~8-12 hours (GPU)

## Files in This Directory

- `glycol.npz` - Original dataset (5,904 structures) ⚠️ Don't use
- `glycol_cleaned.npz` - Cleaned dataset (5,895 structures) ✅ Use this!
- `CLEANING_REPORT.md` - Details about the cleaning process
- `TRAINING_QUICKSTART.md` - Training guide
- `READY_TO_TRAIN.md` - This file

## Getting Help

If you encounter issues:

1. Check that your environment has JAX and mmml installed
2. Verify GPU is available: `python -c "import jax; print(jax.devices())"`
3. Try a smaller test run first
4. Check the logs in your checkpoint directory

## Ready? Let's Train! 🚀

```bash
cd /home/ericb/mmml/examples/glycol

python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_run1 \
  --n_train 4000 \
  --n_valid 800 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --features 128 \
  --num_iterations 3 \
  --cutoff 10.0 \
  --ckpt_dir checkpoints/glycol_run1
```

Everything is configured and ready to go!

