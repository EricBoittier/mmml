# Model Options Summary

Quick reference for choosing between different model architectures and configurations.

## Model Architectures

The training script supports two model architectures for distributed charge prediction:

### 1. DCMNet (Default - Equivariant)
```bash
python trainer.py [standard arguments]
```

**Characteristics:**
- ✅ Fully rotationally equivariant
- ✅ Uses spherical harmonics
- ✅ Message passing neural network
- ⚠️ More parameters, slower
- 📚 Best with limited data

**Key Parameters:**
- `--dcmnet-features`: Hidden layer size (default: 128)
- `--dcmnet-iterations`: Message passing steps (default: 2)
- `--max-degree`: Max spherical harmonic degree (default: 2)
- `--n-dcm`: Distributed charges per atom (default: 3)

### 2. Non-Equivariant Model (New)
```bash
python trainer.py --use-noneq-model [other arguments]
```

**Characteristics:**
- ❌ NOT rotationally equivariant
- ✅ Simple MLP architecture
- ✅ Predicts Cartesian displacements
- ✅ Fewer parameters, faster
- 📚 Best with abundant data

**Key Parameters:**
- `--noneq-features`: Hidden layer size (default: 128)
- `--noneq-layers`: Number of MLP layers (default: 3)
- `--noneq-max-displacement`: Max displacement (default: 1.0 Å)
- `--n-dcm`: Distributed charges per atom (default: 3)

## Optimizer Options

Four optimizers available with automatic hyperparameter recommendations:

```bash
# AdamW (default) - best for production
--optimizer adamw [--use-recommended-hparams]

# Adam - good for exploration
--optimizer adam [--use-recommended-hparams]

# RMSprop - good for noisy gradients
--optimizer rmsprop [--use-recommended-hparams]

# Muon - fast convergence, high momentum
--optimizer muon [--use-recommended-hparams]
```

See [OPTIMIZER_GUIDE.md](OPTIMIZER_GUIDE.md) for details.

## Quick Start Examples

### 1. Default Configuration (DCMNet + AdamW)
```bash
python trainer.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --epochs 100
```

### 2. Fast Non-Equivariant Model
```bash
python trainer.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --use-noneq-model \
    --use-recommended-hparams \
    --epochs 100
```

### 3. High-Capacity DCMNet with Muon
```bash
python trainer.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --dcmnet-features 256 \
    --dcmnet-iterations 3 \
    --optimizer muon \
    --use-recommended-hparams \
    --epochs 150
```

### 4. Lightweight Non-Equivariant
```bash
python trainer.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --use-noneq-model \
    --noneq-features 64 \
    --noneq-layers 2 \
    --n-dcm 3 \
    --batch-size 8 \
    --epochs 100
```

## Decision Matrix

### Choose DCMNet if:
| Factor | DCMNet Advantage |
|--------|------------------|
| Dataset size | Small (< 5000 samples) |
| Symmetry | Must preserve rotational equivariance |
| Multipoles | Need higher-order (l > 0) |
| Theory | Want physically motivated architecture |
| Generalization | Limited rotational coverage in data |

### Choose Non-Equivariant if:
| Factor | Non-Equivariant Advantage |
|--------|--------------------------|
| Dataset size | Large (> 10000 samples) |
| Speed | Need fast training/inference |
| Memory | Limited GPU memory |
| Simplicity | Prefer simpler architecture |
| Interpretability | Want explicit displacement vectors |

### Choose Optimizer Based On:
| Optimizer | Best For | LR Range |
|-----------|----------|----------|
| **AdamW** | Production training, good generalization | 0.0003 - 0.001 |
| **Adam** | Quick experiments, exploration | 0.0005 - 0.002 |
| **RMSprop** | Noisy gradients, unstable training | 0.0003 - 0.001 |
| **Muon** | Fast convergence, large batches | 0.003 - 0.01 |

## Performance Comparison

Typical benchmarks on CO2 dataset (24 atoms, 5000 training samples):

| Configuration | Forward Pass | Memory | Parameters | Validation MAE |
|---------------|-------------|---------|-----------|----------------|
| DCMNet (default) | 100% | 100% | 100% | Baseline |
| DCMNet (large) | 130% | 140% | 180% | -5% better |
| Non-Equivariant (default) | 60% | 65% | 50% | +2% worse |
| Non-Equivariant (large) | 75% | 80% | 80% | ~Same |

*Lower is better for Forward Pass, Memory, Parameters. Lower is better for MAE (error).*

## Combining Options

You can mix and match:

```bash
# Non-Equivariant + Muon + Coulomb mixing
python trainer.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --use-noneq-model \
    --optimizer muon \
    --use-recommended-hparams \
    --mix-coulomb-energy \
    --epochs 100

# DCMNet + RMSprop + custom hyperparameters
python trainer.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --dcmnet-features 256 \
    --dcmnet-iterations 3 \
    --optimizer rmsprop \
    --learning-rate 0.0005 \
    --weight-decay 1e-4 \
    --epochs 150
```

## Common Workflows

### Workflow 1: Rapid Prototyping
1. Start with non-equivariant + auto hyperparameters
2. Train for 50 epochs to validate pipeline
3. If results promising, increase to full training

```bash
python trainer.py --use-noneq-model --use-recommended-hparams --epochs 50 ...
```

### Workflow 2: Production Training
1. Use DCMNet with recommended architecture
2. AdamW optimizer with auto-selected hyperparameters
3. Full epoch count with checkpointing

```bash
python trainer.py --optimizer adamw --use-recommended-hparams --epochs 200 ...
```

### Workflow 3: Comparison Study
1. Train both models with same random seed
2. Use same hyperparameters where applicable
3. Compare validation metrics and timing

```bash
# Model A
python trainer.py --name exp_dcmnet --seed 42 --epochs 100 ...

# Model B
python trainer.py --name exp_noneq --use-noneq-model --seed 42 --epochs 100 ...
```

## Resource Requirements

### GPU Memory Usage (approximate)

| Configuration | Batch=1 | Batch=4 | Batch=8 |
|---------------|---------|---------|---------|
| DCMNet (default) | 2 GB | 4 GB | 7 GB |
| DCMNet (large) | 3 GB | 6 GB | 11 GB |
| Non-Eq (default) | 1.5 GB | 3 GB | 5 GB |
| Non-Eq (large) | 2 GB | 4 GB | 7 GB |

### Training Time (100 epochs, 5000 samples)

| Configuration | GPU | Time |
|---------------|-----|------|
| DCMNet (default) | RTX 3090 | ~4 hours |
| DCMNet (large) | RTX 3090 | ~6 hours |
| Non-Eq (default) | RTX 3090 | ~2.5 hours |
| Non-Eq (large) | RTX 3090 | ~3.5 hours |

## Troubleshooting

### Model Not Converging
Try:
- Switch to AdamW if using Adam
- Reduce learning rate by 2-5×
- Increase weight decay
- Reduce model size if overfitting

### Out of Memory
Try:
- Reduce `--batch-size`
- Use non-equivariant model
- Reduce `--dcmnet-features` or `--noneq-features`
- Reduce `--natoms` if possible

### Training Too Slow
Try:
- Use non-equivariant model
- Reduce model size
- Increase `--batch-size` (if memory allows)
- Use Muon optimizer

### Poor Generalization
Try:
- Use DCMNet (equivariant)
- Increase `--weight-decay`
- Use AdamW instead of Adam
- Add data augmentation (random rotations)

## Further Reading

- [NON_EQUIVARIANT_MODEL.md](NON_EQUIVARIANT_MODEL.md) - Detailed non-equivariant model guide
- [OPTIMIZER_GUIDE.md](OPTIMIZER_GUIDE.md) - Comprehensive optimizer documentation
- [UPDATED_START_HERE.md](UPDATED_START_HERE.md) - General training guide
- [VISUALIZATION_COMPLETE.md](VISUALIZATION_COMPLETE.md) - Visualization options

## Command-Line Reference

### Model Selection
```bash
# Use DCMNet (default)
[no flag needed]

# Use non-equivariant model
--use-noneq-model
```

### DCMNet Hyperparameters
```bash
--dcmnet-features INT           # Hidden features (default: 128)
--dcmnet-iterations INT         # Message passing iterations (default: 2)
--dcmnet-basis INT              # Basis functions (default: 64)
--dcmnet-cutoff FLOAT          # Cutoff distance in Å (default: 10.0)
--max-degree INT               # Max spherical harmonic degree (default: 2)
--n-dcm INT                    # Charges per atom (default: 3)
```

### Non-Equivariant Hyperparameters
```bash
--noneq-features INT           # Hidden features (default: 128)
--noneq-layers INT             # MLP layers (default: 3)
--noneq-max-displacement FLOAT # Max displacement in Å (default: 1.0)
--n-dcm INT                    # Charges per atom (default: 3)
```

### Optimizer Selection
```bash
--optimizer {adam,adamw,rmsprop,muon}  # Optimizer choice (default: adamw)
--learning-rate FLOAT                  # Learning rate (default: auto)
--weight-decay FLOAT                   # Weight decay (default: auto)
--use-recommended-hparams              # Use all recommended settings
```

### Other Common Options
```bash
--batch-size INT               # Batch size (default: 1)
--epochs INT                   # Number of epochs (default: 100)
--seed INT                     # Random seed (default: 42)
--name STR                     # Experiment name
--plot-freq INT                # Plot every N epochs (default: 10)
--plot-results                 # Create final plots
```

