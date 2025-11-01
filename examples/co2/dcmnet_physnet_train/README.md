# DCMNet Training for CO2 ESP Prediction

This directory contains a training script for DCMNet models on CO2 electrostatic potential (ESP) data.

## Overview

DCMNet is a neural network model that predicts electrostatic potentials using:
- **Distributed multipoles** (monopoles and dipoles)
- **Message passing** on molecular graphs
- **Equivariant features** for physical correctness

It's specifically designed for accurate ESP prediction on Van der Waals surfaces.

## Data Preparation

First, prepare your data using the CLI tool:

```bash
cd ..
python fix_and_split_cli.py \
    --efd /path/to/energies_forces_dipoles.npz \
    --grid /path/to/grids_esp.npz \
    --output-dir ./preclassified_data
```

This creates train/valid/test splits with correct units.

## Basic Usage

```bash
python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz
```

## Advanced Usage

### Custom Hyperparameters

```bash
python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz \
    --features 64 \
    --n-dcm 4 \
    --batch-size 16 \
    --epochs 500
```

### Vary Number of Multipoles

```bash
# Train with 1 multipole per atom (monopoles only)
python trainer.py ... --n-dcm 1 --name co2_dcm1

# Train with 3 multipoles per atom
python trainer.py ... --n-dcm 3 --name co2_dcm3

# Train with 5 multipoles per atom (more accurate)
python trainer.py ... --n-dcm 5 --name co2_dcm5
```

### Resume Training

```bash
python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz \
    --restart checkpoints/co2_dcmnet_epoch_50.pkl \
    --name co2_dcmnet_resumed
```

## Command-Line Options

### Data Options
- `--train-efd` - Training EFD NPZ file (required)
- `--train-grid` - Training grid NPZ file (required)
- `--valid-efd` - Validation EFD NPZ file (required)
- `--valid-grid` - Validation grid NPZ file (required)

### Model Hyperparameters
- `--features` (default: 32) - Features per atom
- `--max-degree` (default: 2) - Max spherical harmonic degree
- `--num-iterations` (default: 2) - Message passing iterations
- `--num-basis-functions` (default: 32) - Radial basis functions
- `--cutoff` (default: 10.0) - Cutoff distance in Angstroms
- `--n-dcm` (default: 3) - Distributed multipoles per atom
- `--include-pseudotensors` - Include pseudotensor features

### Training Hyperparameters
- `--batch-size` (default: 32) - Batch size
- `--epochs` (default: 100) - Number of epochs
- `--learning-rate` (default: 0.001) - Learning rate
- `--esp-weight` (default: 10000.0) - ESP loss weight
- `--seed` (default: 42) - Random seed

### Training Options
- `--restart` - Path to restart checkpoint
- `--name` (default: 'co2_dcmnet') - Experiment name
- `--output-dir` (default: './checkpoints') - Checkpoint directory
- `--print-freq` (default: 10) - Print frequency in batches
- `--save-freq` (default: 5) - Save frequency in epochs
- `--verbose` - Verbose output

## Example Training Sessions

### Small Model (Fast Training)

```bash
python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz \
    --features 16 \
    --n-dcm 2 \
    --batch-size 64 \
    --epochs 50 \
    --name co2_dcm2_small
```

### Large Model (High Accuracy)

```bash
python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz \
    --features 64 \
    --n-dcm 5 \
    --num-iterations 3 \
    --batch-size 16 \
    --epochs 1000 \
    --name co2_dcm5_large
```

### Train Multiple Models (Ensemble)

```bash
# Train DCM1-DCM5 for ensemble prediction
for n_dcm in 1 2 3 4 5; do
    python trainer.py \
        --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
        --train-grid ../preclassified_data/grids_esp_train.npz \
        --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
        --valid-grid ../preclassified_data/grids_esp_valid.npz \
        --n-dcm $n_dcm \
        --name co2_dcm${n_dcm} \
        --epochs 200
done
```

## Output

Training creates the following outputs:

### Checkpoints Directory

```
checkpoints/
├── co2_dcmnet_epoch_5.pkl
├── co2_dcmnet_epoch_10.pkl
├── ...
└── co2_dcmnet_final.pkl     # Final model parameters
```

### Training Logs

Progress is printed to console showing:
- Epoch number
- Training loss (ESP MAE)
- Validation loss
- Time per epoch

## Using Trained Models

```python
import pickle
import jax.numpy as jnp
from mmml.dcmnet.dcmnet.modules import MessagePassingModel
from mmml.dcmnet.dcmnet.electrostatics import calc_esp

# Load parameters
with open('checkpoints/co2_dcmnet_final.pkl', 'rb') as f:
    params = pickle.load(f)

# Create model (same hyperparameters as training)
model = MessagePassingModel(
    features=32,
    max_degree=2,
    num_iterations=2,
    num_basis_functions=32,
    cutoff=10.0,
    n_dcm=3,
    include_pseudotensors=False,
)

# Predict multipoles
Z = jnp.array(...)  # Atomic numbers
R = jnp.array(...)  # Coordinates [Angstrom]
dst_idx = jnp.array(...)  # Edge destination indices
src_idx = jnp.array(...)  # Edge source indices

mono, dipo = model.apply(params, Z, R, dst_idx, src_idx)

# mono: Monopoles [e] - shape (n_atoms * n_dcm,)
# dipo: Dipoles [e*Angstrom] - shape (n_atoms * n_dcm, 3)

# Calculate ESP on surface points
vdw_surface = jnp.array(...)  # VdW surface points [Angstrom]
esp_pred = calc_esp(mono, dipo, R, vdw_surface)

# esp_pred: ESP values [Hartree/e] - shape (n_surface_points,)
```

## Ensemble Predictions

For best accuracy, use an ensemble of models:

```python
from mmml.dcmnet.dcmnet.models import (
    DCM1, DCM2, DCM3, DCM4,
    dcm1_params, dcm2_params, dcm3_params, dcm4_params
)

# Use pre-trained models
models = [DCM1, DCM2, DCM3, DCM4]
params_list = [dcm1_params, dcm2_params, dcm3_params, dcm4_params]

# Predict with ensemble (average)
esp_predictions = []
for model, params in zip(models, params_list):
    mono, dipo = model.apply(params, Z, R, dst_idx, src_idx)
    esp = calc_esp(mono, dipo, R, vdw_surface)
    esp_predictions.append(esp)

esp_ensemble = jnp.mean(jnp.stack(esp_predictions), axis=0)
```

## Choosing n_dcm (Number of Multipoles)

- **n_dcm=1**: Monopoles only, fastest, least accurate
- **n_dcm=2**: Good balance of speed and accuracy
- **n_dcm=3**: Recommended default for most applications
- **n_dcm=4-5**: Higher accuracy, slower training
- **n_dcm>5**: Diminishing returns, risk of overfitting

## Requirements

- JAX (with GPU support recommended)
- mmml package (DCMnet submodule)
- NumPy

## Tips

1. **GPU Memory**: Reduce `--batch-size` if you run out of GPU memory
2. **ESP Weight**: Higher values (10000-100000) emphasize ESP accuracy
3. **Multiple Models**: Train different n_dcm values and ensemble them
4. **Convergence**: ESP loss should decrease steadily; if not, adjust learning rate
5. **Cutoff**: Use larger cutoff (10-12 Å) for ESP prediction accuracy

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python trainer.py ... --batch-size 8

# Reduce model size
python trainer.py ... --features 16 --n-dcm 2
```

### Slow Training

```bash
# Increase batch size (if memory allows)
python trainer.py ... --batch-size 64

# Reduce model complexity
python trainer.py ... --num-iterations 1
```

### Poor ESP Prediction

```bash
# Increase number of multipoles
python trainer.py ... --n-dcm 5

# Increase ESP weight
python trainer.py ... --esp-weight 100000.0

# Increase cutoff
python trainer.py ... --cutoff 12.0
```

## Performance Metrics

Expected ESP MAE (Mean Absolute Error):
- **Monopoles only (n_dcm=1)**: ~0.01-0.02 Hartree/e
- **n_dcm=2**: ~0.005-0.01 Hartree/e  
- **n_dcm=3**: ~0.003-0.008 Hartree/e
- **n_dcm=4-5**: ~0.002-0.005 Hartree/e

(These are approximate and depend on molecule complexity and training data)

## Citation

If you use DCMNet, please cite:

```bibtex
@article{veit2024dcmnet,
  title={DCMNet: Distributed Charge and Multipole Network for molecular property prediction},
  author={Veit, Michael and others},
  journal={...},
  year={2024}
}
```

## Support

For issues or questions:
- Check the main MMML documentation
- Open an issue on the MMML repository
- Review training logs for diagnostic information

