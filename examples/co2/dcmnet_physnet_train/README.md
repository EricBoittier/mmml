# Joint PhysNet-DCMNet Training

This directory contains a joint training script that trains PhysNet and DCMNet simultaneously with end-to-end gradient flow.

## Architecture

1. **PhysNet** predicts atomic charges (supervised by molecular dipole D)
2. Those charges become **monopoles** for DCMNet
3. **DCMNet** predicts distributed multipoles for ESP fitting
4. Full gradient flow: ESP loss → DCMNet → charges → PhysNet

## Usage

Basic usage:
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.00001 \
  --grad-clip-norm 1.0
```

Recommended settings for stable training:
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.00001 \
  --energy-weight 1.0 \
  --forces-weight 1.0 \
  --dipole-weight 1.0 \
  --esp-weight 1.0 \
  --mono-weight 0.1 \
  --grad-clip-norm 1.0 \
  --name co2_joint_stable
```

## Performance Notes

### Speed Optimization

The current implementation is ~140s/epoch for 8000 samples with batch_size=1. The main bottleneck is:

1. **Edge list construction in Python loops** - The `prepare_batch_data()` function constructs edge lists for each batch using nested Python loops
2. **Batch overhead** - With batch_size=1, there are 8000 batches per epoch

**Potential speedups:**

1. **Pre-compute edge lists once before training** - Store them with the data
2. **Increase batch size** - Requires handling variable numbers of ESP grid points per molecule
3. **Vectorize edge construction** - Use JAX operations instead of Python loops
4. **Cache edge lists** - Store computed edge lists between epochs

### Gradient Clipping

Gradient clipping is enabled by default (`--grad-clip-norm 1.0`) to prevent exploding gradients. Disable with `--grad-clip-norm 0` if not needed.

## Loss Components

- **Energy**: PhysNet energy prediction (eV)
- **Forces**: PhysNet force prediction (eV/Å)
- **Dipole**: PhysNet dipole prediction from charges (Debye)
- **ESP**: DCMNet electrostatic potential fitting (Hartree/e)
- **Monopole**: Constraint that sum of DCMNet distributed charges = PhysNet charge per atom

## Hyperparameters

### PhysNet
- `--physnet-features`: Number of features (default: 64)
- `--physnet-iterations`: Message passing iterations (default: 5)
- `--physnet-basis`: Radial basis functions (default: 64)
- `--physnet-cutoff`: Interaction cutoff in Å (default: 6.0)

### DCMNet
- `--dcmnet-features`: Number of features (default: 32)
- `--dcmnet-iterations`: Message passing iterations (default: 2)
- `--dcmnet-basis`: Radial basis functions (default: 32)
- `--dcmnet-cutoff`: Interaction cutoff in Å (default: 10.0)
- `--n-dcm`: Distributed multipoles per atom (default: 3)

### Loss Weights
Start with balanced weights (all 1.0) and adjust based on loss magnitudes:
- `--energy-weight`: Energy loss weight (default: 1.0)
- `--forces-weight`: Forces loss weight (default: 50.0)
- `--dipole-weight`: Dipole loss weight (default: 25.0)
- `--esp-weight`: ESP loss weight (default: 10000.0)
- `--mono-weight`: Monopole constraint weight (default: 1.0)
