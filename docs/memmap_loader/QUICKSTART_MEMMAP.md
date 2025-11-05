# Quick Start: PhysNet Training with Packed Memmap Data

Get started training PhysNet on large molecular datasets in 5 minutes.

## Prerequisites

- JAX with GPU support
- Your molecular data in packed memmap format
- See `PACKED_MEMMAP_TRAINING.md` for detailed data format specification

## Quick Start

### 1. Verify Your Data

Make sure your data directory has the required files:

```bash
ls -lh your_data_path/
# Should show:
# offsets.npy, n_atoms.npy, Z_pack.int32, R_pack.f32, F_pack.f32, E.f64, Qtot.f64
```

### 2. Run Simple Example

Test with the minimal example (10 epochs):

```bash
cd /home/ericb/mmml
python examples/train_memmap_simple.py
```

Edit the script to point to your data:
```python
DATA_PATH = "your_data_path"  # Change this
```

### 3. Full Training

Run the full training script:

```bash
python train_physnet_memmap.py \
    --data_path your_data_path \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --num_atoms 60 \
    --name my_experiment
```

That's it! Your model will train and save checkpoints.

## Common Configurations

### Small Molecules (< 30 atoms, e.g., QM9)

```bash
python train_physnet_memmap.py \
    --data_path qm9_data \
    --batch_size 128 \
    --num_atoms 29 \
    --bucket_size 4096 \
    --features 128 \
    --cutoff 5.0
```

### Medium Molecules (30-60 atoms, e.g., SPICE)

```bash
python train_physnet_memmap.py \
    --data_path spice_data \
    --batch_size 64 \
    --num_atoms 60 \
    --bucket_size 8192 \
    --features 256 \
    --cutoff 5.0
```

### Large Molecules (> 60 atoms, e.g., Peptides)

```bash
python train_physnet_memmap.py \
    --data_path peptide_data \
    --batch_size 32 \
    --num_atoms 100 \
    --bucket_size 16384 \
    --features 256 \
    --cutoff 6.0
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--data_path` | Path to packed memmap directory | (required) |
| `--batch_size` | Molecules per batch | 32-128 |
| `--num_atoms` | Max atoms (padding size) | 29-100 |
| `--num_epochs` | Training epochs | 100-500 |
| `--learning_rate` | Learning rate | 0.0001-0.001 |
| `--features` | Hidden layer size | 64-256 |
| `--cutoff` | Interaction cutoff (Å) | 5.0-6.0 |
| `--bucket_size` | Bucketing group size | 4096-16384 |

## Expected Output

```
================================================================================
PhysNet Training with Packed Memmap Data
================================================================================
Data path: your_data_path
Batch size: 32
Number of epochs: 100
Learning rate: 0.001
================================================================================

Loading data...
Total molecules: 100000
Training molecules: 90000
Validation molecules: 10000

Initializing model...
Model initialized with 1,234,567 parameters

Starting training...
================================================================================

Epoch 1/100
--------------------------------------------------------------------------------
Training...
  Batch 0: Loss=5.432100, E_MAE=12.345000, F_MAE=8.901000
  Batch 50: Loss=4.123000, E_MAE=9.876000, F_MAE=6.543000
  ...
Validating...

Epoch 1 Results:
  Train Loss: 3.456789
  Train Energy MAE: 8.234567 kcal/mol
  Train Forces MAE: 5.678901 kcal/mol/Å
  Valid Loss: 3.234567
  Valid Energy MAE: 7.890123 kcal/mol
  Valid Forces MAE: 5.432109 kcal/mol/Å
  Time: 45.67 s
  → New best validation loss! Saving checkpoint...
```

## Troubleshooting

### "Out of memory" error
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--num_atoms`
- Use smaller model: `--features 64`

### Training is slow
- Increase `--batch_size` to saturate GPU
- Ensure data is on SSD/NVMe storage
- Check GPU utilization: `nvidia-smi`

### "File size mismatch" error
- Verify data files are complete
- Regenerate packed data files
- Check `offsets[-1]` matches total atoms

## Next Steps

1. **Monitor training**: Checkpoints saved to `physnetjax/ckpt/`
2. **Adjust hyperparameters**: See `PACKED_MEMMAP_TRAINING.md`
3. **Load checkpoint**: Use `mmml.physnetjax.restart.restart_training()`
4. **Evaluate model**: Load EMA params and run inference

## Python API

For custom training loops:

```python
from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader
from mmml.physnetjax.physnetjax.models.model import EF

# Load data
loader = PackedMemmapLoader("your_data_path", batch_size=32)
train_loader, valid_loader = split_loader(loader, train_fraction=0.9)

# Create model
model = EF(features=128, num_iterations=3, natoms=60)

# Train
for batch in train_loader.batches(num_atoms=60):
    # batch: dict with keys Z, R, F, E, N, Qtot, dst_idx, src_idx, batch_segments
    # Your training step here
    pass
```

## Full Documentation

See `PACKED_MEMMAP_TRAINING.md` for:
- Detailed data format specification
- Advanced usage examples
- Performance optimization tips
- Complete API reference
- Troubleshooting guide

## Getting Help

If you encounter issues:
1. Check `PACKED_MEMMAP_TRAINING.md` documentation
2. Run the simple example: `python examples/train_memmap_simple.py`
3. Verify data format matches specification
4. Check GPU memory: `nvidia-smi`
5. Try smaller batch size and simpler model first

