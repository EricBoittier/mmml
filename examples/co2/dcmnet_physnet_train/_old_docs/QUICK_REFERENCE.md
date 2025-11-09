# Quick Reference Card

One-page summary of all training options.

## Model Architectures

| Model | Flag | Equivariant? | Speed | Params |
|-------|------|-------------|-------|--------|
| **DCMNet** | _(default)_ | ✅ Yes | Normal | More |
| **Non-Equivariant** | `--use-noneq-model` | ❌ No | ~2× faster | Fewer |

## Optimizers

| Optimizer | Flag | Best For | LR Range |
|-----------|------|----------|----------|
| **AdamW** | `--optimizer adamw` | Production (default) | 0.0003-0.001 |
| **Adam** | `--optimizer adam` | Exploration | 0.0005-0.002 |
| **RMSprop** | `--optimizer rmsprop` | Noisy gradients | 0.0003-0.001 |
| **Muon** | `--optimizer muon` | Fast convergence | 0.003-0.01 |

## Common Command Patterns

### Minimal (defaults)
```bash
python trainer.py \
  --train-efd train.npz --train-esp esp_train.npz \
  --valid-efd valid.npz --valid-esp esp_valid.npz
```

### Fast non-equivariant
```bash
python trainer.py ... --use-noneq-model
```

### Auto hyperparameters
```bash
python trainer.py ... --use-recommended-hparams
```

### Different optimizer
```bash
python trainer.py ... --optimizer muon
```

### Everything auto-tuned
```bash
python trainer.py ... --use-noneq-model --optimizer muon --use-recommended-hparams
```

## Key Hyperparameters

### DCMNet
- `--dcmnet-features 128` - Hidden size
- `--dcmnet-iterations 2` - Message passing steps
- `--max-degree 2` - Spherical harmonic degree
- `--n-dcm 3` - Charges per atom

### Non-Equivariant
- `--noneq-features 128` - Hidden size
- `--noneq-layers 3` - MLP depth
- `--noneq-max-displacement 1.0` - Max distance (Å)
- `--n-dcm 3` - Charges per atom

### Training
- `--batch-size 1` - Batch size
- `--epochs 100` - Number of epochs
- `--learning-rate` - LR (auto if not set)
- `--weight-decay` - Weight decay (auto if not set)
- `--seed 42` - Random seed

## Preset Configurations

### Small molecules, fast iteration
```bash
--use-noneq-model \
--noneq-features 64 --noneq-layers 2 \
--batch-size 8 --epochs 50
```

### Production training
```bash
--optimizer adamw --use-recommended-hparams \
--batch-size 4 --epochs 200
```

### Large molecules, high accuracy
```bash
--dcmnet-features 256 --dcmnet-iterations 3 \
--max-degree 3 --n-dcm 5 \
--optimizer muon --use-recommended-hparams
```

### Comparison study
```bash
# Run twice with different --name:
--name exp1_dcmnet --seed 42
--name exp2_noneq --use-noneq-model --seed 42
```

## Files Generated

- `checkpoints/{name}/best_params.pkl` - Best model weights
- `checkpoints/{name}/history.json` - Training metrics
- `checkpoints/{name}/plots/` - Validation plots (if `--plot-freq > 0`)

## Documentation

- `MODEL_OPTIONS.md` - Complete model comparison
- `NON_EQUIVARIANT_MODEL.md` - Non-equivariant details
- `OPTIMIZER_GUIDE.md` - Optimizer documentation
- `UPDATED_START_HERE.md` - Full training guide

## Decision Tree

```
Start
 ├─ Need guaranteed equivariance? 
 │   ├─ Yes → Use DCMNet (default)
 │   └─ No → Go to next question
 ├─ Have large dataset (>10k)? 
 │   ├─ Yes → Use Non-Equivariant (--use-noneq-model)
 │   └─ No → Use DCMNet (default)
 ├─ Need fast training? 
 │   ├─ Yes → Use Non-Equivariant (--use-noneq-model)
 │   └─ No → Use DCMNet (default)
 └─ Optimizer choice:
     ├─ Production → AdamW (default)
     ├─ Fast convergence → Muon (--optimizer muon)
     ├─ Exploration → Adam (--optimizer adam)
     └─ Unstable → RMSprop (--optimizer rmsprop)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` or use `--use-noneq-model` |
| Not converging | Use `--optimizer adamw`, reduce LR, increase weight decay |
| Too slow | Use `--use-noneq-model` or increase `--batch-size` |
| Poor validation | Use DCMNet, increase `--weight-decay`, more epochs |
| NaN loss | Reduce LR, check data, enable `--grad-clip-norm 1.0` |

## Example Workflows

### Rapid prototyping (30 mins)
```bash
python trainer.py \
  --train-efd train.npz --train-esp esp_train.npz \
  --valid-efd valid.npz --valid-esp esp_valid.npz \
  --use-noneq-model --noneq-features 64 --noneq-layers 2 \
  --batch-size 8 --epochs 30 --name quick_test
```

### Production run (4-6 hours)
```bash
python trainer.py \
  --train-efd train.npz --train-esp esp_train.npz \
  --valid-efd valid.npz --valid-esp esp_valid.npz \
  --optimizer adamw --use-recommended-hparams \
  --batch-size 4 --epochs 200 --name production_v1 \
  --plot-freq 10 --plot-results
```

### Comparison A/B test
```bash
# Test A: DCMNet
python trainer.py ... --name test_dcmnet --seed 42 --epochs 100

# Test B: Non-Equivariant  
python trainer.py ... --name test_noneq --use-noneq-model --seed 42 --epochs 100

# Compare: checkpoints/test_*/history.json
```

