# Minimal PhysNet Training Example - Glycol Dataset

This directory contains a minimal example for training PhysNet on the glycol.npz dataset for energies and forces prediction.

## Files

- `glycol.npz` - Dataset with 5,904 molecular configurations (energies and forces)
- `train_minimal.py` - Minimal training script (full 50 epochs)
- `test_training.py` - Quick training test (1 batch to verify setup)
- `test_energy_preprocessing.py` - Quick data inspection script

## Dataset Statistics

- Total molecules: 5,904
- Max atoms per molecule: 10
- Average atoms per molecule: ~10.0
- Energy range: [-228.5, 0.0] eV
- Forces in eV/Å

**Note**: The forces in this dataset have very large magnitudes (up to ~10^12 eV/Å), which may indicate numerical issues in the original data. The training script will still work, but you may want to investigate or preprocess the forces data if accuracy is critical.

## Quick Start

### 1. Test Data Loading

```bash
python test_energy_preprocessing.py
```

### 2. Verify Training Setup (Optional but Recommended)

```bash
python test_training.py
```

This runs a quick test with 1 batch to verify everything works.

### 3. Run Full Training

```bash
python train_minimal.py
```

The training script will:
- Split data into train/valid/test (80%/10%/10%)
- Train for 50 epochs
- Print loss, energy MAE, and forces MAE for each epoch
- Report final test set performance

## Configuration

Default settings in `train_minimal.py`:
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50
- Model: 128 features, 3 iterations, 3 residual blocks

To modify, edit the configuration section in `main()`.

### Optional: Enable JAX 64-bit Mode

To enable float64 precision (removes dtype warning):

```bash
export JAX_ENABLE_X64=1
python train_minimal.py
```

Or set it in the script before imports:
```python
import os
os.environ['JAX_ENABLE_X64'] = '1'
```

## Expected Output

Training output will show per-epoch metrics:
```
Epoch 1/50
Train Loss: X.XXXXXX | E_MAE: X.XXXX eV | F_MAE: X.XXXX eV/Å
Valid Loss: X.XXXXXX | E_MAE: X.XXXX eV | F_MAE: X.XXXX eV/Å
```

