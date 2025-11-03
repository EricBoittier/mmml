# What's New - Model Comparison & Equivariance Testing

## Summary

This update adds comprehensive tools for comparing equivariant (DCMNet) and non-equivariant models, including automated comparison scripts and visual demonstrations of equivariance properties.

## New Scripts

### 1. compare_models.py
**Full head-to-head model comparison**

```bash
python compare_models.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --epochs 50 --comparison-name my_test
```

**Features:**
- ✅ Trains both models with identical settings
- ✅ Automated equivariance testing (rotations + translations)
- ✅ Performance benchmarking (speed, memory, accuracy)
- ✅ Generates comparison plots automatically
- ✅ Saves detailed JSON results

**Output:**
```
comparisons/my_test/
├── comparison_results.json          # All metrics
├── performance_comparison.png       # Accuracy plots
├── efficiency_comparison.png        # Speed/memory/parameters
├── equivariance_comparison.png      # Rotation/translation tests
├── dcmnet_equivariant/              # DCMNet checkpoint
└── noneq_model/                     # Non-Eq checkpoint
```

### 2. demo_equivariance.py
**Visual equivariance demonstration**

```bash
python demo_equivariance.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --test-data valid.npz --test-esp grids_valid.npz \
    --rotation-angle 90 --rotation-axis z
```

**Features:**
- ✅ Clear demonstration of equivariance concept
- ✅ Applies rotations and translations
- ✅ Shows prediction errors for both models
- ✅ Creates 3D visualizations
- ✅ Educational tool for understanding equivariance

**Example Output:**
```
ROTATION EQUIVARIANCE (90° around z):
  DCMNet:     Error = 0.000002 e·Å  ✅ EQUIVARIANT
  Non-Eq:     Error = 0.145000 e·Å  ⚠️  NOT EQUIVARIANT (expected)

TRANSLATION INVARIANCE ([5.0, 3.0, -2.0] Å):
  DCMNet:     Error = 0.000003 e·Å  ✅ INVARIANT
  Non-Eq:     Error = 0.000004 e·Å  ✅ INVARIANT
```

## New Documentation

### 1. COMPARISON_GUIDE.md
Complete guide to using comparison tools:
- Detailed explanation of equivariance tests
- How rotation/translation tests work
- Expected results and interpretation
- Example workflows
- Troubleshooting

### 2. README.md
Comprehensive overview:
- Quick links to all documentation
- Summary of all features
- Example workflows
- Performance comparisons
- Citation information

### 3. WHATS_NEW.md (this file)
Summary of recent additions

## Key Concepts

### Equivariance Test

**What it tests:** Does model output rotate correctly when input rotates?

**Method:**
1. Predict for original molecule → `dipole_1`
2. Rotate molecule by angle θ → `molecule_rotated`
3. Predict for rotated molecule → `dipole_2`
4. Check: `dipole_2 ≈ R(θ) @ dipole_1`?

**Results:**
- **DCMNet**: Error ≈ 10⁻⁶ (perfect equivariance)
- **Non-Eq**: Error ≈ 0.1-0.3 (not equivariant)

### Translation Test

**What it tests:** Is output invariant to translating molecule in space?

**Method:**
1. Predict for original molecule → `output_1`
2. Translate molecule by vector Δr → `molecule_translated`
3. Predict for translated molecule → `output_2`
4. Check: `output_2 ≈ output_1`?

**Results:**
- **Both models**: Error ≈ 10⁻⁶ (perfect invariance)

## Use Cases

### 1. Educational

Demonstrate equivariance to students/colleagues:
```bash
# Visual demonstration
python demo_equivariance.py \
    --checkpoint-dcm model1.pkl \
    --checkpoint-noneq model2.pkl \
    --test-data test.npz --test-esp grids.npz \
    --rotation-angle 45
```

### 2. Research

Compare models for publication:
```bash
# Full comparison with statistics
python compare_models.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --epochs 100 --equivariance-samples 100 \
    --comparison-name paper_comparison
```

### 3. Model Selection

Decide which model to use:
```bash
# Quick comparison
python compare_models.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --epochs 30 --comparison-name quick_test

# Review results/plots to make decision
```

### 4. Debugging

Verify equivariance is working:
```bash
# Test a trained DCMNet model
python demo_equivariance.py \
    --checkpoint-dcm trained_model.pkl \
    --checkpoint-noneq baseline.pkl \
    --test-data test.npz --test-esp grids.npz

# Should see DCMNet error ≈ 10⁻⁶
# If not, there's a bug!
```

## Integration with Existing Features

All new features work seamlessly with existing capabilities:

✅ **Multiple optimizers** (Adam, AdamW, RMSprop, Muon)
✅ **Auto hyperparameter tuning**
✅ **Loss configuration**
✅ **Visualization tools**
✅ **Checkpoint management**
✅ **EMA validation**

## Example Workflow

### Complete Model Comparison Study

```bash
# Step 1: Compare models
python compare_models.py \
    --train-efd train.npz --train-esp grids_train.npz \
    --valid-efd valid.npz --valid-esp grids_valid.npz \
    --epochs 100 --comparison-name study1 \
    --equivariance-samples 50

# Step 2: Visualize equivariance
python demo_equivariance.py \
    --checkpoint-dcm comparisons/study1/dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq comparisons/study1/noneq_model/best_params.pkl \
    --test-data valid.npz --test-esp grids_valid.npz \
    --rotation-angle 90 --rotation-axis z \
    --output-dir comparisons/study1/demo

# Step 3: Review results
cd comparisons/study1/
open performance_comparison.png
open equivariance_comparison.png
open demo/equivariance_demo.png
cat comparison_results.json
```

## What The Plots Show

### performance_comparison.png
4 subplots showing validation MAE:
- Energy (eV)
- Forces (eV/Å)
- Dipole (e·Å)
- ESP (Ha/e)

Bar chart comparison between DCMNet and Non-Eq.

### efficiency_comparison.png
3 subplots showing:
- Training time (hours)
- Inference time (milliseconds)
- Parameter count (millions)

Shows computational advantages of Non-Eq.

### equivariance_comparison.png
2 subplots:
- Rotation errors (log scale)
- Translation errors (log scale)

Clearly demonstrates:
- DCMNet: Near-zero rotation error (equivariant)
- Non-Eq: Large rotation error (not equivariant)
- Both: Near-zero translation error (invariant)

### equivariance_demo.png (from demo_equivariance.py)
6 subplots in 3D:
- Top row: DCMNet predictions (original, rotated, translated)
- Bottom row: Non-Eq predictions (original, rotated, translated)

Visual demonstration of dipole vectors under transformations.

## Technical Details

### Rotation Test Implementation

```python
# 1. Generate random rotation matrix
R = random_rotation_matrix()

# 2. Rotate molecule
positions_rotated = R @ positions

# 3. Get predictions
dipole_original = predict(positions)
dipole_rotated = predict(positions_rotated)

# 4. Check equivariance
dipole_expected = R @ dipole_original
error = ||dipole_rotated - dipole_expected||

# DCMNet: error ≈ 10⁻⁶ ✅
# Non-Eq:  error ≈ 0.1-0.3 ⚠️
```

### Translation Test Implementation

```python
# 1. Random translation vector
t = random_vector()

# 2. Translate molecule
positions_translated = positions + t

# 3. Get predictions
output_original = predict(positions)
output_translated = predict(positions_translated)

# 4. Check invariance
error = ||output_translated - output_original||

# Both models: error ≈ 10⁻⁶ ✅
```

## Command Cheat Sheet

### Run full comparison
```bash
python compare_models.py \
  --train-efd train.npz --train-esp grids_train.npz \
  --valid-efd valid.npz --valid-esp grids_valid.npz \
  --epochs 50 --comparison-name test1
```

### Demo equivariance
```bash
python demo_equivariance.py \
  --checkpoint-dcm model1.pkl --checkpoint-noneq model2.pkl \
  --test-data valid.npz --test-esp grids_valid.npz
```

### Skip training, reuse checkpoints
```bash
python compare_models.py ... --skip-training
```

### More equivariance samples
```bash
python compare_models.py ... --equivariance-samples 100
```

### Different rotation
```bash
python demo_equivariance.py ... --rotation-angle 45 --rotation-axis y
```

## Files Added/Modified

### New Files
- `compare_models.py` - Comparison script
- `demo_equivariance.py` - Equivariance demo
- `COMPARISON_GUIDE.md` - Comparison documentation
- `README.md` - Main readme
- `WHATS_NEW.md` - This file

### Modified Files
- `trainer.py` - Already updated (no changes needed)

### Existing Documentation
- `MODEL_OPTIONS.md` - Already created
- `NON_EQUIVARIANT_MODEL.md` - Already created
- `OPTIMIZER_GUIDE.md` - Already created
- `QUICK_REFERENCE.md` - Already created

## Questions & Answers

**Q: How long does a comparison take?**
A: With 50 epochs: 2-4 hours depending on hardware.

**Q: Can I use existing checkpoints?**
A: Yes! Use `--skip-training` flag.

**Q: What if I only have one model trained?**
A: Use `demo_equivariance.py` - it works with any trained checkpoints.

**Q: How many samples for equivariance testing?**
A: Default 20 is usually sufficient. Use 100+ for publication.

**Q: Does this work with custom loss configurations?**
A: Yes! Both scripts use the same training pipeline.

**Q: Can I compare different hyperparameters?**
A: `compare_models.py` uses same hyperparameters for fair comparison. For different settings, train separately with `trainer.py`.

## Next Steps

1. **Try the demo:**
   ```bash
   python demo_equivariance.py --help
   ```

2. **Run a quick comparison:**
   ```bash
   python compare_models.py --epochs 20 --comparison-name quick_test [data args]
   ```

3. **Read the guides:**
   - `COMPARISON_GUIDE.md` - Detailed usage
   - `README.md` - Complete overview

4. **Integrate into your workflow:**
   - Use for model selection
   - Include in papers/presentations
   - Teach equivariance concepts

## Support

See documentation files for detailed information:
- `COMPARISON_GUIDE.md` - Complete comparison guide
- `README.md` - Overview and quick start
- `MODEL_OPTIONS.md` - Model architecture details
- `QUICK_REFERENCE.md` - One-page cheat sheet

