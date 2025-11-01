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

### Speed Optimization ✅

**Edge lists are now pre-computed once before training!** This provides a massive speedup:

- **Before**: ~140s/epoch (edge lists computed 8000 times per epoch)
- **After**: ~10-20s/epoch expected (edge lists computed once at startup)

The `precompute_edge_lists()` function:
1. Computes edge lists for all 8000 training samples once (~10 seconds)
2. Stores them as object arrays in the dataset
3. `prepare_batch_data()` just extracts and concatenates pre-computed edges

**Additional speedup options:**
1. **Increase batch size** - Requires handling variable numbers of ESP grid points per molecule
2. **Use compiled edge construction** - Vectorize with JAX/NumPy operations

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
