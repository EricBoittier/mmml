# Glycol Training Quickstart

Your glycol dataset has been cleaned and is ready for training! 🎉

## Quick Start (Recommended)

The simplest command - `num_atoms` is now auto-detected:

```bash
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_run1 \
  --n_train 3000 \
  --n_valid 500 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --ckpt_dir checkpoints/glycol_run1
```

**What changed:** You no longer need to specify `--num_atoms 60` - it's automatically detected from your dataset!

## What Happens

The training script will:
1. ✅ **Auto-detect num_atoms = 60** from `glycol_cleaned.npz` 
2. Load 3000 training samples and 500 validation samples
3. Create a PhysNet model with default architecture
4. Train for 50 epochs
5. Save checkpoints to `checkpoints/glycol_run1/`

## Output

You'll see:
```
Auto-detecting number of atoms from dataset...
  ✅ Detected num_atoms = 60 from R.shape (includes padding)
     (Note: Actual molecule size = 10, padded to 60)
```

## Dataset Info

- **File:** `glycol_cleaned.npz`
- **Structures:** 4,130 (cleaned from 5,904)
- **Actual atoms:** 10 (glycol molecule: C₂H₆O₂)
- **Padded to:** 60 atoms (for batching)
- **Training will use:** 60 (the padded size) ✅

## Advanced Options

### Custom Model Architecture

```bash
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_custom \
  --n_train 3000 \
  --n_valid 500 \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --features 128 \
  --num_iterations 3 \
  --num_basis_functions 64 \
  --cutoff 10.0 \
  --ckpt_dir checkpoints/glycol_custom
```

### Resume Training

```bash
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_run1_cont \
  --restart checkpoints/glycol_run1/checkpoint.pkl \
  --num_epochs 50 \
  --ckpt_dir checkpoints/glycol_run1_cont
```

## Monitor Training

After starting training, you can monitor progress:

```bash
# Plot training curves (real-time)
python -m mmml.cli.plot_training checkpoints/glycol_run1/history.json \
  --output-dir plots
```

## After Training

### Test the Model

```bash
# Test with calculator
python -m mmml.cli.calculator \
  --checkpoint checkpoints/glycol_run1 \
  --test-molecule H2O
```

### Run Dynamics

```bash
# Geometry optimization
python -m mmml.cli.dynamics \
  --checkpoint checkpoints/glycol_run1 \
  --molecule CO2 \
  --optimize \
  --output-dir glycol_dynamics
```

## Troubleshooting

### If auto-detection fails
Specify `--num_atoms` explicitly:
```bash
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --num_atoms 60 \
  ...
```

### If training is slow
- Increase `--batch_size` (try 16 or 32)
- Reduce model size: `--features 64 --num_iterations 2`

### If training diverges
- Reduce `--learning_rate` (try 0.0001)
- Increase `--energy_weight` for better energy fitting

## Next Steps

1. ✅ **Start training** with the command above
2. 📊 **Monitor** with `plot_training.py`
3. 🧪 **Test** with `calculator.py`
4. ⚛️ **Analyze** with `dynamics.py`

Good luck with your training! 🚀

