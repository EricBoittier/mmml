# Migration Guide: dc.py → Hydra-based Training

This guide shows how your original `dc.py` script maps to the new Hydra-based structure.

## What Changed?

### Before (dc.py)
- Single monolithic script (~308 lines)
- Hard-coded parameters
- Difficult to run experiments with different settings
- No organized output structure
- Manual parameter tracking

### After (Hydra Structure)
- Modular, reusable code
- Configuration-driven experiments
- Easy hyperparameter sweeps
- Organized outputs with timestamps
- Optional experiment tracking (W&B)

## File Mapping

| Original (dc.py) | New Structure | Description |
|-----------------|---------------|-------------|
| Lines 1-15 | `configs/config.yaml` | Device and environment setup |
| Lines 17-35 | `configs/model/base.yaml` | Model configuration |
| Lines 38-51 | `configs/data/water.yaml` | Data paths and loading |
| Lines 52-132 | `utils.py:prepare_data()` | Data preparation functions |
| Lines 65-84 | `utils.py:random_sample_esp()` | ESP sampling |
| Lines 138-146 | `train.py:main()` | Training loop |
| Lines 149-170 | `evaluate.py` | Analysis and evaluation |
| Lines 173-292 | `evaluate.py` | Visualization functions |

## Configuration Equivalents

### Original Hard-coded Values
```python
# dc.py
NDCM = 4
features = 128
max_degree = 2
num_iterations = 2
num_basis_functions = 32
cutoff = 8.0
```

### New Config File
```yaml
# configs/model/base.yaml
model:
  n_dcm: 4
  features: 128
  max_degree: 2
  num_iterations: 2
  num_basis_functions: 32
  cutoff: 8.0
```

### Original Training Loop
```python
# dc.py
for i in range(Nboot):
    params, valid_loss = train_model(
        key=data_key, model=model,
        train_data=train_data, valid_data=valid_data,
        num_epochs=50, learning_rate=1e-4, batch_size=1,
        esp_w=1000.0*((i+1)/Nboot), chg_w=1.0/((i+1)),
    )
```

### New Config-Driven Training
```yaml
# configs/training/bootstrap.yaml
training:
  num_epochs: 50
  learning_rate: 1.0e-4
  n_bootstrap: 10
  esp_weight_schedule: "linear"
  chg_weight_schedule: "inverse"
```

## Running Equivalent Experiments

### Original: Edit code and run
```bash
# Edit dc.py to change parameters
# Then run:
python dc.py
```

### New: Override configs
```bash
# Run with different parameters (no code editing!)
python train.py model.n_dcm=6 training.learning_rate=5e-4

# Or use different config groups
python train.py model=large training=standard

# Run multiple experiments
python train.py -m model.features=64,128,256 seed=42,43,44
```

## Key Advantages

### 1. Reproducibility
**Before:** 
- No record of exact parameters used
- Hard to reproduce results

**After:**
- Full config saved with each run in `outputs/YYYY-MM-DD/HH-MM-SS/`
- Can reproduce exactly: `python train.py --config-path /path/to/outputs/.hydra/`

### 2. Experiment Organization
**Before:**
- Outputs overwrite each other
- Manual tracking of results

**After:**
- Each run gets timestamped directory
- All outputs organized together
- Easy to compare multiple runs

### 3. Hyperparameter Sweeps
**Before:**
- Write loops manually
- Track results by hand

**After:**
```bash
# Automatic sweep over 27 combinations
python train.py -m \
    model.features=64,128,256 \
    training.learning_rate=1e-4,5e-4,1e-3 \
    seed=42,43,44
```

### 4. Smart Optimization
**Before:**
- Manual trial and error

**After:**
```bash
# Optuna tries 50 smart combinations
python train.py -m --config-name=sweep_optuna
```

## Quick Reference

| Task | Old Command | New Command |
|------|-------------|-------------|
| Basic training | `python dc.py` | `python train.py` |
| Change model size | Edit line 31 | `python train.py model.features=256` |
| Change learning rate | Edit line 142 | `python train.py training.learning_rate=5e-4` |
| Change bootstrap iterations | Edit line 53 | `python train.py training.n_bootstrap=20` |
| Run multiple seeds | Edit and run multiple times | `python train.py -m seed=42,43,44` |
| Evaluate | Lines 164-292 | `python evaluate.py checkpoint_path=...` |

## Backward Compatibility

Your original `dc.py` is preserved and still works! The new structure is purely additive.

## Next Steps

1. **Try the quick test**: `python train.py experiment=quick_test`
2. **Run a sweep**: `python train.py -m model.features=64,128 seed=42,43`
3. **Evaluate results**: `python evaluate.py checkpoint_path=outputs/.../final_params.npz`
4. **Customize configs**: Edit `configs/` files for your experiments

## Need Help?

- Read `README.md` for detailed examples
- Run `./run_examples.sh` for interactive examples
- Check `configs/experiment/quick_test.yaml` for a minimal example

