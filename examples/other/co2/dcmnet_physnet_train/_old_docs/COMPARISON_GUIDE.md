# Model Comparison Guide

This guide explains how to use `compare_models.py` to perform head-to-head comparisons between the equivariant (DCMNet) and non-equivariant models.

## What Does It Do?

The comparison script:

1. **Trains both models** with identical hyperparameters for fair comparison
2. **Tests equivariance** by applying random rotations and translations
3. **Benchmarks performance** including training time, inference speed, and memory
4. **Generates detailed reports** with visualizations and metrics
5. **Demonstrates the difference** between equivariant and non-equivariant architectures

## Quick Start

### Basic Usage

```bash
python compare_models.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --epochs 50 \
    --comparison-name my_first_comparison
```

This will:
- Train both models for 50 epochs
- Run equivariance tests on 20 validation samples
- Generate comparison plots
- Save all results to `comparisons/my_first_comparison/`

### Output Files

After running, you'll find:

```
comparisons/my_first_comparison/
├── comparison_results.json          # Detailed metrics
├── performance_comparison.png       # Validation MAE comparison
├── efficiency_comparison.png        # Training time, inference, parameters
├── equivariance_comparison.png      # Rotation and translation tests
├── dcmnet_equivariant/              # DCMNet checkpoint
│   ├── best_params.pkl
│   └── history.json
└── noneq_model/                     # Non-Eq checkpoint
    ├── best_params.pkl
    └── history.json
```

## Understanding the Tests

### 1. Equivariance Test (Rotation)

**What it tests**: Does the model's output rotate correctly when the input is rotated?

**How it works**:
1. Predict dipole and ESP for original molecule
2. Rotate molecule by random 3D rotation
3. Predict dipole and ESP for rotated molecule
4. Check if rotated prediction equals rotation of original prediction

**Expected results**:
- **DCMNet (Equivariant)**: Near-zero error (~1e-6 or less)
  - Dipole should transform as: `dipole_rotated = R @ dipole_original`
  - ESP should be identical at corresponding grid points
  
- **Non-Equivariant**: Larger error (depends on training data)
  - Model must learn rotational behavior from data
  - Error shows lack of guaranteed equivariance

**Example output**:
```
Rotation Equivariance:
  DCMNet:
    Dipole error: 0.000002 ± 0.000001 e·Å  ✅ Perfect!
    ESP error:    0.000015 ± 0.000008 Ha/e  ✅ Perfect!
  
  Non-Equivariant:
    Dipole error: 0.145300 ± 0.082000 e·Å  ⚠️ Not equivariant
    ESP error:    0.002340 ± 0.001200 Ha/e  ⚠️ Not equivariant
```

### 2. Translation Invariance Test

**What it tests**: Is the output invariant to translating the molecule in space?

**How it works**:
1. Predict for original molecule
2. Translate molecule by random vector (5 Å scale)
3. Predict for translated molecule
4. Check if predictions are identical

**Expected results**:
- **Both models**: Near-zero error
  - Both use internal/COM-relative coordinates
  - Translation should not affect predictions
  
**Example output**:
```
Translation Invariance:
  DCMNet:
    Dipole error: 0.000003 ± 0.000002 e·Å  ✅ Perfect!
    ESP error:    0.000018 ± 0.000012 Ha/e  ✅ Perfect!
  
  Non-Equivariant:
    Dipole error: 0.000004 ± 0.000002 e·Å  ✅ Perfect!
    ESP error:    0.000021 ± 0.000015 Ha/e  ✅ Perfect!
```

## Command-Line Options

### Required Arguments

```bash
--train-efd PATH         # Training EFD data
--train-esp PATH         # Training ESP data
--valid-efd PATH         # Validation EFD data
--valid-esp PATH         # Validation ESP data
```

### Training Options

```bash
--epochs INT             # Number of epochs (default: 50)
--batch-size INT         # Batch size (default: 4)
--seed INT               # Random seed (default: 42)
```

### Model Configuration

```bash
--physnet-features INT   # PhysNet hidden size (default: 64)
--dcmnet-features INT    # DCMNet/Non-Eq hidden size (default: 128)
--n-dcm INT              # Charges per atom (default: 3)
```

### Comparison Options

```bash
--comparison-name STR           # Name for results (default: model_comparison)
--output-dir PATH               # Output directory (default: comparisons/)
--skip-training                 # Load existing checkpoints, skip training
--equivariance-samples INT      # Samples for equivariance tests (default: 20)
```

## Use Cases

### 1. Quick Comparison (30 minutes)

Fast comparison with minimal epochs:

```bash
python compare_models.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --epochs 30 \
    --batch-size 8 \
    --physnet-features 64 \
    --dcmnet-features 64 \
    --comparison-name quick_test
```

### 2. Production Comparison (4-6 hours)

Full comparison with sufficient training:

```bash
python compare_models.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --epochs 100 \
    --batch-size 4 \
    --physnet-features 64 \
    --dcmnet-features 128 \
    --comparison-name production_comparison
```

### 3. Detailed Equivariance Study

Focus on equivariance testing:

```bash
python compare_models.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --epochs 50 \
    --equivariance-samples 100 \
    --comparison-name equivariance_study
```

### 4. Re-analyze Existing Models

Skip training and just run tests on existing checkpoints:

```bash
python compare_models.py \
    --train-efd train.npz \
    --train-esp grids_train.npz \
    --valid-efd valid.npz \
    --valid-esp grids_valid.npz \
    --skip-training \
    --comparison-name previous_comparison
```

## Interpreting Results

### Performance Metrics

The script reports validation MAE for:

- **Energy**: eV per molecule
- **Forces**: eV/Å per atom
- **Dipole**: e·Å (electron·Angstrom)
- **ESP**: Ha/e (Hartree per electron)

**Lower is better** for all metrics.

**Expected patterns**:
- Similar accuracy for both models with sufficient data
- DCMNet may perform better with limited data
- Non-Eq may match or exceed DCMNet with abundant data

### Efficiency Metrics

**Training Time**:
- Non-Eq typically 1.5-2× faster
- Due to simpler architecture (no message passing)

**Inference Time**:
- Non-Eq typically 1.5-2× faster
- Important for production deployment

**Parameters**:
- Non-Eq typically 30-50% fewer parameters
- Affects memory usage and model size

### Equivariance Errors

**Rotation Errors**:

| Model | Expected Range | Interpretation |
|-------|---------------|----------------|
| DCMNet | 1e-6 to 1e-4 | Perfect equivariance (numerical precision) |
| Non-Eq | 0.01 to 0.5 | Learned approximate equivariance |

**Translation Errors**:

| Model | Expected Range | Interpretation |
|-------|---------------|----------------|
| Both | 1e-6 to 1e-4 | Perfect invariance |

**What if Non-Eq has large rotation error?**
- Expected behavior! It's not designed to be equivariant
- Can be reduced with:
  - More training data with diverse rotations
  - Data augmentation
  - More model capacity
  
**What if DCMNet has large rotation error?**
- Something is wrong! Check:
  - Implementation bugs
  - Numerical issues
  - Model configuration

## Visualizations

### 1. Performance Comparison Plot

Shows validation MAE for all four metrics side-by-side.

**What to look for**:
- Are the bars similar height? (similar accuracy)
- Which model is better for each metric?
- Is the difference significant?

### 2. Efficiency Comparison Plot

Shows training time, inference time, and parameter count.

**What to look for**:
- How much faster is Non-Eq?
- Is the speedup worth any accuracy loss?
- Memory/parameter savings

### 3. Equivariance Comparison Plot

Shows rotation and translation errors (log scale).

**What to look for**:
- DCMNet bars should be near the bottom (near-zero)
- Non-Eq rotation bars will be higher (not equivariant)
- Both translation bars should be near-zero

**Key insight**: This plot visually demonstrates the difference between equivariant and non-equivariant architectures!

## Example Workflow

### Step 1: Run Comparison

```bash
python compare_models.py \
    --train-efd data/train.npz \
    --train-esp data/grids_train.npz \
    --valid-efd data/valid.npz \
    --valid-esp data/grids_valid.npz \
    --epochs 50 \
    --comparison-name co2_comparison \
    --seed 42
```

### Step 2: Check Results

```bash
cd comparisons/co2_comparison/
cat comparison_results.json
```

Look for:
- Validation MAE values
- Rotation error difference (DCMNet << Non-Eq expected)
- Training time ratio
- Parameter count ratio

### Step 3: View Plots

Open the three PNG files:
- `performance_comparison.png` - Accuracy comparison
- `efficiency_comparison.png` - Speed and size comparison  
- `equivariance_comparison.png` - Symmetry test results

### Step 4: Interpret

**If Non-Eq has similar accuracy**:
- ✅ Good choice for production (faster, smaller)
- ✅ Sufficient training data
- Consider Non-Eq for deployment

**If DCMNet has better accuracy**:
- ✅ Benefit from equivariance
- Limited training data or rotational diversity
- Use DCMNet for best accuracy

**Equivariance matters for your application?**
- Yes → Use DCMNet (guaranteed equivariance)
- No → Either model works, consider efficiency

## Advanced Usage

### Custom Training Configuration

You can modify the script to use different:
- Optimizers
- Learning rates
- Loss weights
- Model architectures

Edit the script around line 700 where models are created.

### Multiple Comparisons

Run multiple comparisons with different settings:

```bash
# Small model
python compare_models.py ... --dcmnet-features 64 --comparison-name small_model

# Large model  
python compare_models.py ... --dcmnet-features 256 --comparison-name large_model

# Many charges per atom
python compare_models.py ... --n-dcm 5 --comparison-name n_dcm_5
```

### Batch Processing

Compare across multiple datasets:

```bash
for dataset in dataset1 dataset2 dataset3; do
    python compare_models.py \
        --train-efd ${dataset}/train.npz \
        --train-esp ${dataset}/grids_train.npz \
        --valid-efd ${dataset}/valid.npz \
        --valid-esp ${dataset}/grids_valid.npz \
        --comparison-name comparison_${dataset} \
        --epochs 50
done
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python compare_models.py ... --batch-size 2

# Reduce model size
python compare_models.py ... --physnet-features 32 --dcmnet-features 64
```

### Training Takes Too Long

```bash
# Reduce epochs for quick test
python compare_models.py ... --epochs 20

# Or load existing checkpoints
python compare_models.py ... --skip-training
```

### Unexpected Results

**DCMNet not equivariant?**
- Check implementation
- Verify data preprocessing (no fixed frame?)
- Check numerical precision

**Non-Eq has near-zero rotation error?**
- May have learned perfect equivariance from data!
- Possible if training data has comprehensive rotational coverage

**Both models have poor accuracy?**
- Insufficient training epochs
- Wrong hyperparameters
- Data quality issues

## Tips for Best Results

1. **Use same random seed** for reproducibility
2. **Train for sufficient epochs** (50-100 minimum)
3. **Test on held-out data** not seen during training
4. **Multiple random seeds** for statistical significance
5. **Document results** with comparison names

## Integration with Main Training

The comparison script uses the same codebase as `trainer.py`, so:

✅ **Configurations are compatible**
- Can load checkpoints from `trainer.py`
- Can continue training with `trainer.py`

✅ **Results are comparable**
- Same evaluation metrics
- Same data preprocessing

✅ **Easy to extend**
- Add custom tests
- Modify plots
- Add new metrics

## Citation and References

When publishing results using this comparison:

```bibtex
@software{model_comparison,
  title = {Equivariant vs Non-Equivariant Model Comparison},
  year = {2025},
  note = {Head-to-head comparison of DCMNet and Non-Equivariant charge models}
}
```

## Further Reading

- `MODEL_OPTIONS.md` - Overview of model architectures
- `NON_EQUIVARIANT_MODEL.md` - Details on non-equivariant model
- `UPDATED_START_HERE.md` - General training guide

## Quick Reference

| Task | Command |
|------|---------|
| Basic comparison | `python compare_models.py --train-efd ... --epochs 50` |
| Quick test | `python compare_models.py ... --epochs 20 --batch-size 8` |
| Skip training | `python compare_models.py ... --skip-training` |
| More equivariance tests | `python compare_models.py ... --equivariance-samples 100` |
| Custom name | `python compare_models.py ... --comparison-name my_test` |

