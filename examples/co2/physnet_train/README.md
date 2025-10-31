# PhysNetJax Training for CO2

This directory contains a training script for PhysNetJax models on CO2 molecular data.

## Overview

PhysNetJax is a neural network model that predicts:
- **Energies** (eV)
- **Forces** (eV/Angstrom)
- **Dipoles** (Debye)

It uses message passing on molecular graphs with equivariant features.

## Data Preparation

First, prepare your data using the CLI tool:

```bash
cd ..
python fix_and_split_cli.py \
    --efd /path/to/energies_forces_dipoles.npz \
    --grid /path/to/grids_esp.npz \
    --output-dir ./preclassified_data
```

This creates train/valid/test splits with correct ASE units.

## Basic Usage

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz
```

## Advanced Usage

### Custom Hyperparameters

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --features 128 \
    --num-iterations 5 \
    --cutoff 8.0 \
    --batch-size 16 \
    --epochs 500
```

### Custom Loss Weights

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --energy-weight 1.0 \
    --forces-weight 100.0 \
    --dipole-weight 50.0
```

### Resume Training

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --restart \
    --name co2_physnet_resumed
```

## Command-Line Options

### Data Options
- `--train` - Path to training NPZ file (required)
- `--valid` - Path to validation NPZ file (required)

### Model Hyperparameters
- `--features` (default: 64) - Features per atom
- `--max-degree` (default: 3) - Max spherical harmonic degree
- `--num-iterations` (default: 3) - Message passing iterations
- `--num-basis-functions` (default: 32) - Radial basis functions
- `--cutoff` (default: 6.0) - Cutoff distance in Angstroms
- `--n-res` (default: 3) - Number of residual blocks
- `--natoms` (default: 60) - Maximum atoms per molecule
- `--zbl` - Use ZBL repulsion (default: True)
- `--charges` - Predict atomic charges

### Training Hyperparameters
- `--batch-size` (default: 32) - Batch size
- `--epochs` (default: 100) - Maximum epochs
- `--learning-rate` (default: 0.001) - Learning rate
- `--seed` (default: 42) - Random seed

### Loss Weights
- `--energy-weight` (default: 1.0) - Energy loss weight
- `--forces-weight` (default: 50.0) - Forces loss weight
- `--dipole-weight` (default: 25.0) - Dipole loss weight

### Training Options
- `--restart` - Restart from checkpoint
- `--name` (default: 'co2_physnet') - Experiment name
- `--ckpt-dir` - Checkpoint directory
- `--print-freq` (default: 1) - Print frequency in epochs
- `--no-tensorboard` - Disable TensorBoard logging
- `--objective` (default: 'valid_forces_mae') - Early stopping metric
  - Options: valid_forces_mae, valid_energy_mae, valid_loss, train_forces_mae, train_energy_mae, train_loss

### Optimizer Options
- `--optimizer` (default: 'amsgrad') - Optimizer type
  - Options: adam, adamw, amsgrad
- `--schedule` (default: 'warmup') - Learning rate schedule
  - Options: warmup, cosine_annealing, exponential, polynomial, cosine, warmup_cosine, constant
- `--transform` (default: 'reduce_on_plateau') - LR transform

### Preprocessing Options
- `--center-coordinates` - Center coordinates at origin
- `--normalize-energy` - Normalize energies
- `--verbose` - Verbose output

## Example Training Sessions

### Small Model (Fast Training)

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --features 32 \
    --num-iterations 2 \
    --batch-size 64 \
    --epochs 50 \
    --name co2_physnet_small
```

### Large Model (High Accuracy)

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --features 128 \
    --num-iterations 5 \
    --num-basis-functions 64 \
    --cutoff 8.0 \
    --batch-size 16 \
    --epochs 1000 \
    --name co2_physnet_large
```

### Energy-Focused Training

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --energy-weight 10.0 \
    --forces-weight 10.0 \
    --dipole-weight 5.0 \
    --objective valid_energy_mae \
    --name co2_physnet_energy
```

### Forces-Focused Training

```bash
python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --energy-weight 1.0 \
    --forces-weight 100.0 \
    --dipole-weight 25.0 \
    --objective valid_forces_mae \
    --name co2_physnet_forces
```

## Output

Training creates the following outputs:

### Checkpoints Directory

```
checkpoints/co2_physnet/
├── best_params.pkl          # Best model parameters
├── checkpoint_epoch_*.pkl   # Periodic checkpoints
└── training_history.json    # Training metrics
```

### TensorBoard Logs

```
runs/co2_physnet/
└── events.out.tfevents.*    # TensorBoard logs
```

View with: `tensorboard --logdir runs/`

## Using Trained Models

```python
import pickle
import jax.numpy as jnp
from mmml.physnetjax.physnetjax.models.model import EF

# Load parameters
with open('checkpoints/co2_physnet/best_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Create model (same hyperparameters as training)
model = EF(
    features=64,
    max_degree=3,
    num_iterations=3,
    num_basis_functions=32,
    cutoff=6.0,
    max_atomic_number=118,
    charges=False,
    natoms=60,
    total_charge=0.0,
    n_res=3,
    zbl=True,
    debug=False,
    efa=False,
)

# Predict
R = jnp.array(...)  # Coordinates [Angstrom]
Z = jnp.array(...)  # Atomic numbers
dst_idx = jnp.array(...)  # Edge destination indices
src_idx = jnp.array(...)  # Edge source indices

E, F, D = model.apply(params, R, Z, dst_idx, src_idx, ...)

# E: Energy [eV]
# F: Forces [eV/Angstrom]
# D: Dipole [Debye]
```

## Requirements

- JAX (with GPU support recommended)
- mmml package (PhysNetJax submodule)
- NumPy
- TensorBoard (optional, for logging)

## Tips

1. **GPU Memory**: Reduce `--batch-size` if you run out of GPU memory
2. **Training Speed**: Increase `--batch-size` for faster training (if memory allows)
3. **Convergence**: Monitor TensorBoard to check convergence
4. **Hyperparameters**: Start with defaults, then tune based on validation metrics
5. **Early Stopping**: Training will save the best model based on `--objective`

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python trainer.py ... --batch-size 8
```

### Slow Training

```bash
# Smaller model
python trainer.py ... --features 32 --num-iterations 2

# Larger batch size (if memory allows)
python trainer.py ... --batch-size 64
```

### Poor Convergence

```bash
# Adjust loss weights
python trainer.py ... --forces-weight 100.0

# Try different optimizer
python trainer.py ... --optimizer adam --schedule cosine_annealing
```

## Citation

If you use PhysNetJax, please cite:

```bibtex
@article{unke2019physnet,
  title={PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges},
  author={Unke, Oliver T and Meuwly, Markus},
  journal={Journal of chemical theory and computation},
  volume={15},
  number={6},
  pages={3678--3693},
  year={2019},
  publisher={ACS Publications}
}
```

## Support

For issues or questions:
- Check the main MMML documentation
- Open an issue on the MMML repository
- Review TensorBoard logs for training diagnostics

