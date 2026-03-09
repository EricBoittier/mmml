# Optimizer Support Guide

The training script now supports multiple optimizers with automatic hyperparameter recommendations based on dataset properties.

## Supported Optimizers

1. **Adam** - Adaptive Moment Estimation (no weight decay)
2. **AdamW** - Adam with decoupled weight decay (default)
3. **RMSprop** - Root Mean Square Propagation
4. **Muon** - Momentum Orthogonalized by Newton's method (approximated with heavy ball SGD)

## Usage

### Basic Usage

Select an optimizer with the `--optimizer` flag:

```bash
python trainer.py \
    --train-efd energies_forces_dipoles_train.npz \
    --train-esp grids_esp_train.npz \
    --valid-efd energies_forces_dipoles_valid.npz \
    --valid-esp grids_esp_valid.npz \
    --optimizer adamw  # Options: adam, adamw, rmsprop, muon
```

### Auto-Selected Hyperparameters

By default, if you don't specify `--learning-rate` or `--weight-decay`, the script will automatically select recommended values based on:
- Dataset size (number of training samples)
- Model complexity (total features: PhysNet + DCMNet)
- Number of atoms

Example with auto-selection:
```bash
python trainer.py \
    --train-efd data_train.npz \
    --train-esp grids_train.npz \
    --valid-efd data_valid.npz \
    --valid-esp grids_valid.npz \
    --optimizer rmsprop
    # Learning rate and weight decay will be auto-selected
```

### Use Full Recommended Configuration

To use all recommended hyperparameters (including optimizer-specific parameters like momentum, beta values, etc.):

```bash
python trainer.py \
    --train-efd data_train.npz \
    --train-esp grids_train.npz \
    --valid-efd data_valid.npz \
    --valid-esp grids_valid.npz \
    --optimizer muon \
    --use-recommended-hparams
```

This will print and use all recommended parameters:
```
🔧 Using recommended hyperparameters for MUON:
  learning_rate: 0.01
  momentum: 0.95
  weight_decay: 0.001
  nesterov: True
```

### Manual Configuration

You can still manually specify hyperparameters:

```bash
python trainer.py \
    --train-efd data_train.npz \
    --train-esp grids_train.npz \
    --valid-efd data_valid.npz \
    --valid-esp grids_valid.npz \
    --optimizer adam \
    --learning-rate 0.0005 \
    --weight-decay 0.0
```

## Optimizer Characteristics

### Adam
- **When to use**: Good general-purpose optimizer, works well with sparse gradients
- **Typical LR**: 0.001 (small datasets), 0.0005 (large datasets)
- **Weight decay**: 0.0 (Adam doesn't use weight decay)
- **Best for**: Fast convergence, exploration phase

### AdamW (Default)
- **When to use**: Better generalization than Adam, especially for large models
- **Typical LR**: 0.001 (small datasets), 0.0005 (large datasets)
- **Weight decay**: 1e-4 (standard models), 1e-3 (large models)
- **Best for**: Production training with good generalization

### RMSprop
- **When to use**: Good for recurrent architectures and non-stationary objectives
- **Typical LR**: 0.001 (small datasets), 0.0005 (large datasets)
- **Weight decay**: 1e-5 (standard), 1e-4 (large models)
- **Momentum**: 0.0 (default)
- **Best for**: Training with noisy gradients

### Muon
- **When to use**: When you need fast convergence with high momentum
- **Typical LR**: 0.01 (small datasets), 0.005 (large datasets) - higher than others!
- **Weight decay**: 1e-4 (standard), 1e-3 (large models)
- **Momentum**: 0.95
- **Best for**: Large batch training, fast convergence
- **Note**: Currently implemented as SGD with heavy ball momentum. For the full Muon optimizer, install: `pip install muon-optimizer`

## Hyperparameter Recommendations

The recommendation system uses the following heuristics:

### Small Dataset Detection
- **Threshold**: < 1000 training samples
- **Adjustment**: Higher learning rates for faster convergence

### Large Model Detection
- **Threshold**: > 256 total features OR > 50 atoms
- **Adjustment**: 
  - Reduced learning rates (×0.5-0.7)
  - Increased weight decay for better regularization

### Examples

**Small dataset (500 samples), small model (128 features):**
```
AdamW: lr=0.001, wd=1e-4
Muon:  lr=0.01, wd=1e-4
```

**Large dataset (5000 samples), large model (384 features):**
```
AdamW: lr=0.00025, wd=1e-3  # (0.0005 × 0.5 for large model)
Muon:  lr=0.0025, wd=1e-3   # (0.005 × 0.5 for large model)
```

## Integration with Existing Workflows

### Backward Compatibility
All existing training scripts will continue to work with the default AdamW optimizer and auto-selected hyperparameters.

### Checkpoint Compatibility
Optimizer state is saved in checkpoints. When restarting:
- Use the same optimizer as the original training
- Hyperparameters are loaded from the checkpoint directory

## Advanced Usage

### Custom Optimizer Parameters Dictionary

You can programmatically create configurations:

```python
from trainer import get_recommended_optimizer_config, create_optimizer

# Get recommendations
config = get_recommended_optimizer_config(
    dataset_size=2000,
    num_features=192,
    num_atoms=30,
    optimizer_name='rmsprop'
)

# Create optimizer
optimizer = create_optimizer(
    optimizer_name='rmsprop',
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    decay=config.get('decay', 0.9),
    momentum=config.get('momentum', 0.0)
)
```

### Lambda Functions for Dynamic Configuration

The `OPTIMIZER_CONFIGS` dictionary contains lambda functions for dynamic recommendations:

```python
from trainer import OPTIMIZER_CONFIGS

# Get recommended config for your dataset
recommended = OPTIMIZER_CONFIGS['adamw'](
    ds_size=1500,      # 1500 training samples
    features=256,      # 256 total features
    atoms=24           # 24 max atoms
)
```

## Troubleshooting

### Training Diverges
- Try reducing learning rate by 2-5×
- Increase weight decay
- Switch to AdamW if using Adam
- Ensure gradient clipping is enabled (`--grad-clip-norm 1.0`)

### Slow Convergence
- Try increasing learning rate (especially for Muon)
- Reduce weight decay
- Switch to Muon for faster convergence
- Check if using too small batch size

### Poor Generalization
- Increase weight decay
- Use AdamW instead of Adam
- Reduce learning rate
- Enable EMA (already enabled by default with decay=0.999)

## Examples

### Quick start with defaults:
```bash
python trainer.py \
    --train-efd train.npz --train-esp train_esp.npz \
    --valid-efd valid.npz --valid-esp valid_esp.npz
```

### Try different optimizers:
```bash
# Adam - fast exploration
python trainer.py --optimizer adam --use-recommended-hparams ...

# AdamW - production training (default)
python trainer.py --optimizer adamw --use-recommended-hparams ...

# RMSprop - noisy gradients
python trainer.py --optimizer rmsprop --use-recommended-hparams ...

# Muon - fast convergence
python trainer.py --optimizer muon --use-recommended-hparams ...
```

### Fine-tune manually:
```bash
python trainer.py \
    --optimizer adamw \
    --learning-rate 0.0003 \
    --weight-decay 5e-4 \
    ...
```

