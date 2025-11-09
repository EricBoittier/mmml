# Comprehensive Split Evaluation System

Complete evaluation pipeline for analyzing model performance across train/valid/test splits with geometric feature analysis.

## Overview

This system evaluates your trained model on different data splits and creates comprehensive visualizations showing how errors vary with:
- **1D plots**: Individual geometric features (r1, r2, angle, r1+r2, r1×r2) vs errors
- **2D plots**: Combined features (r1+r2+r1r2 vs angle) vs errors

## Quick Start

```bash
# Run complete evaluation pipeline
./run_full_evaluation.sh \
    --checkpoint ./ckpts/co2_joint_physnet_dcmnet \
    --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
    --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
    --train-esp ../dcmnet_train/grids_esp_train.npz \
    --valid-esp ../dcmnet_train/grids_esp_valid.npz \
    --output-dir ./evaluation_results
```

## Manual Step-by-Step

### Step 1: Python Evaluation

```bash
python evaluate_splits.py \
    --checkpoint ./ckpts/model \
    --train-efd ./data/train_efd.npz \
    --valid-efd ./data/valid_efd.npz \
    --train-esp ./data/train_esp.npz \
    --valid-esp ./data/valid_esp.npz \
    --output-dir ./evaluation
```

**Optional test split:**
```bash
python evaluate_splits.py \
    --checkpoint ./ckpts/model \
    --train-efd ./data/train_efd.npz \
    --valid-efd ./data/valid_efd.npz \
    --test-efd ./data/test_efd.npz \
    --train-esp ./data/train_esp.npz \
    --valid-esp ./data/valid_esp.npz \
    --test-esp ./data/test_esp.npz \
    --output-dir ./evaluation
```

**Output:**
- `evaluation_results.csv` - Complete data with geometric features and errors

### Step 2: R Plotting

```bash
Rscript plot_evaluation_results.R ./evaluation
```

**Output:**
- `plots/` directory with all visualizations

## Generated Plots

### 1D Plots (vs Bonds/Angles)

For each metric, plots vs: **r1**, **r2**, **r1+r2**, **r1×r2**, **angle**

1. **Energy Errors**
   - `energy_error_vs_r1.png`
   - `energy_error_vs_r2.png`
   - `energy_error_vs_r1_plus_r2.png`
   - `energy_error_vs_r1_times_r2.png`
   - `energy_error_vs_angle.png`

2. **Force Norm Errors**
   - `force_norm_error_vs_r1.png`
   - `force_norm_error_vs_r2.png`
   - `force_norm_error_vs_r1_plus_r2.png`
   - `force_norm_error_vs_r1_times_r2.png`
   - `force_norm_error_vs_angle.png`

3. **Force Max Errors**
   - `force_max_error_vs_r1.png`
   - `force_max_error_vs_r2.png`
   - `force_max_error_vs_angle.png`

4. **Dipole Errors (PhysNet)**
   - `dipole_physnet_error_vs_r1.png`
   - `dipole_physnet_error_vs_r2.png`
   - `dipole_physnet_error_vs_angle.png`

5. **Dipole Errors (DCMNet)**
   - `dipole_dcmnet_error_vs_r1.png`
   - `dipole_dcmnet_error_vs_r2.png`
   - `dipole_dcmnet_error_vs_angle.png`

6. **ESP RMSE**
   - `esp_rmse_physnet_vs_r1.png`
   - `esp_rmse_physnet_vs_r2.png`
   - `esp_rmse_physnet_vs_angle.png`
   - `esp_rmse_dcmnet_vs_r1.png`
   - `esp_rmse_dcmnet_vs_r2.png`
   - `esp_rmse_dcmnet_vs_angle.png`

### 2D Plots (r1+r2+r1r2 vs Angle)

Heatmaps showing error distributions:

- `energy_error_2d.png`
- `force_norm_error_2d.png`
- `force_max_error_2d.png`
- `dipole_physnet_error_2d.png`
- `dipole_dcmnet_error_2d.png`
- `esp_rmse_physnet_2d.png`
- `esp_rmse_dcmnet_2d.png`

## Geometric Features Computed

For CO₂ molecule (C-O1-O2):

| Feature | Description | Units |
|---------|-------------|-------|
| `r1` | C-O1 bond length | Å |
| `r2` | C-O2 bond length | Å |
| `angle` | O1-C-O2 angle | degrees |
| `r1_plus_r2` | Sum of bond lengths | Å |
| `r1_times_r2` | Product of bond lengths | Å² |
| `r1_plus_r2_plus_r1r2` | Combined feature for 2D plots | Å + Å² |

## CSV Output Format

The `evaluation_results.csv` contains:

### Geometric Features
- `split`: train/valid/test
- `r1`, `r2`, `angle`, `r1_plus_r2`, `r1_times_r2`

### Energy Metrics
- `energy_true`, `energy_pred`, `energy_error`, `energy_abs_error`

### Force Metrics
- `force_norm_true`, `force_norm_pred`, `force_norm_error`, `force_norm_abs_error`
- `force_max_abs_error`

### Dipole Metrics (PhysNet)
- `dipole_physnet_norm_true`, `dipole_physnet_norm_pred`
- `dipole_physnet_norm_error`, `dipole_physnet_norm_abs_error`

### Dipole Metrics (DCMNet)
- `dipole_dcmnet_norm_true`, `dipole_dcmnet_norm_pred`
- `dipole_dcmnet_norm_error`, `dipole_dcmnet_norm_abs_error`

### ESP Metrics
- `esp_rmse_physnet`, `esp_rmse_dcmnet` (if ESP data available)

## R Requirements

Install required R packages:

```r
install.packages(c("ggplot2", "dplyr", "gridExtra", "viridis"))
```

Or from command line:
```bash
Rscript -e "install.packages(c('ggplot2', 'dplyr', 'gridExtra', 'viridis'), repos='https://cloud.r-project.org')"
```

## Example Usage

### Full Pipeline
```bash
./run_full_evaluation.sh \
    --checkpoint /path/to/ckpt \
    --train-efd /path/to/train_efd.npz \
    --valid-efd /path/to/valid_efd.npz \
    --train-esp /path/to/train_esp.npz \
    --valid-esp /path/to/valid_esp.npz \
    --output-dir ./evaluation_run1
```

### Python Only (if R not available)
```bash
python evaluate_splits.py \
    --checkpoint ./ckpts/model \
    --train-efd ./data/train.npz \
    --valid-efd ./data/valid.npz \
    --train-esp ./data/train_esp.npz \
    --valid-esp ./data/valid_esp.npz \
    --output-dir ./evaluation

# Then manually run R later:
# Rscript plot_evaluation_results.R ./evaluation
```

## Interpreting Results

### 1D Plots
- **High errors at specific r1/r2**: Model struggles with certain bond lengths
- **High errors at specific angles**: Model struggles with bent geometries
- **Smooth vs noisy**: Smooth = systematic error, noisy = random/underfitting

### 2D Plots
- **Red/yellow regions**: Problematic geometry combinations
- **Blue regions**: Well-predicted geometries
- **Gaps**: Underrepresented regions in training data (good for active learning!)

## Troubleshooting

### "Rscript not found"
Install R:
```bash
# Ubuntu/Debian
sudo apt install r-base

# macOS
brew install r
```

### "Package not found" in R
```bash
Rscript -e "install.packages(c('ggplot2', 'dplyr', 'gridExtra', 'viridis'), repos='https://cloud.r-project.org')"
```

### "Memory error" during evaluation
Reduce batch size:
```bash
python evaluate_splits.py ... --batch-size 50
```

### "No ESP data" warnings
ESP plots require ESP training data. Either:
- Provide ESP data files
- Or plots will be skipped (not a critical error)

## Performance

**Typical runtime:**
- Python evaluation: ~5-30 min (depends on dataset size)
- R plotting: ~1-5 min

**Memory:**
- Python: ~2-4 GB
- R: ~1-2 GB

## Integration with Active Learning

The 2D plots reveal **gaps in training data**:
```bash
# 1. Run evaluation
./run_full_evaluation.sh ... --output-dir ./eval_v1

# 2. Identify high-error regions in plots
# 3. Generate geometries for those regions
# 4. Run QM calculations (see ACTIVE_LEARNING_README.md)
# 5. Retrain with new data
# 6. Re-run evaluation to verify improvement
```

## Output Structure

```
evaluation/
├── evaluation_results.csv      # Raw data (all splits)
└── plots/
    ├── energy_error_vs_r1.png
    ├── energy_error_vs_r2.png
    ├── ...
    ├── energy_error_2d.png
    ├── force_norm_error_2d.png
    └── ...
```

## Next Steps

1. ✅ Run evaluation on current model
2. 📊 Analyze plots to identify problematic regions
3. 🎯 Use active learning to target weak regions
4. 🔄 Retrain and re-evaluate
5. 📈 Compare before/after plots

