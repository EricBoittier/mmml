# Training History Plotting Guide

Comprehensive CLI for visualizing training curves, convergence, and parameter structures.

---

## 🚀 Quick Start

### Basic Usage

```bash
# Plot single training run
python plot_training_history.py checkpoints/my_model/history.json

# Compare two runs
python plot_training_history.py \
    dcmnet/history.json noneq/history.json \
    --compare --names "DCMNet" "Non-Eq"

# With parameter analysis
python plot_training_history.py history.json \
    --params best_params.pkl --analyze-params
```

### Complete Analysis

```bash
# Everything: training curves + convergence + parameters
python plot_training_history.py \
    model1/history.json model2/history.json \
    --compare \
    --names "DCMNet" "Non-Eq" \
    --params model1/best_params.pkl model2/best_params.pkl \
    --analyze-params \
    --convergence \
    --smoothing 0.9 \
    --dpi 200
```

---

## 📊 Plot Types

### 1. Training Comparison

**Command:**
```bash
python plot_training_history.py hist1.json hist2.json --compare
```

**Shows:**
- Validation loss over epochs
- Energy, Forces, Dipole, ESP MAE curves
- Training speed (time per epoch)
- Best epoch markers
- Smooth curves with raw data overlay

**File:** `training_comparison.png`

### 2. Convergence Analysis

**Command:**
```bash
python plot_training_history.py history.json --convergence
```

**Shows:**
- Per-epoch improvement rate
- Rolling average improvement (50-epoch window)
- Training progress (% to best model)
- Final validation metrics bar chart
- Convergence point detection

**File:** `convergence_analysis_<model>.png`

### 3. Parameter Analysis

**Command:**
```bash
python plot_training_history.py history.json \
    --params best_params.pkl --analyze-params
```

**Shows:**
- Module parameter counts (pie chart)
- Module parameter counts (bar chart with hatching)
- Layer size distribution (histogram)
- Parameter tree summary (text box)
- Largest layers highlighted

**File:** `parameter_analysis_<model>.png`

###4. Parameter Comparison

**Command:**
```bash
python plot_training_history.py hist1.json hist2.json \
    --compare \
    --params params1.pkl params2.pkl \
    --analyze-params
```

**Shows:**
- Total parameter count comparison
- Module-by-module comparison (grouped bar chart)
- Size difference and percentages
- Hatching patterns to distinguish models

**File:** `parameter_comparison.png`

---

## ⚙️ Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `history_files` | One or more training history JSON files |

### Optional Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--compare` | Compare two training runs | False |
| `--params PATH [PATH ...]` | Parameter pickle files | None |
| `--analyze-params` | Create parameter analysis plots | False |
| `--convergence` | Create convergence analysis plots | False |
| `--output-dir PATH` | Output directory | Same as history |
| `--dpi INT` | Image resolution | 150 |
| `--format {png,pdf,svg,jpg}` | Output format | png |
| `--smoothing FLOAT` | Exponential smoothing (0-1) | 0.0 |
| `--names STR [STR ...]` | Custom model names | Auto |
| `--summary-only` | Only print text, no plots | False |

---

## 📖 Usage Examples

### Example 1: Quick Training Summary

```bash
python plot_training_history.py checkpoints/my_model/history.json --summary-only
```

**Output:**
```
======================================================================
Model - TRAINING SUMMARY
======================================================================

Total Epochs: 1000

Loss:
  Final Train Loss: 0.8088
  Final Val Loss:   0.9914
  Best Val Loss:    0.9914 @ Epoch 1000

Final Validation MAE:
  Energy: 0.123018 eV
  Forces: 0.013880 eV/Å
  Dipole: 0.092861 e·Å
  ESP:    0.004650 Ha/e

Training Speed:
  Avg time per epoch: 2.85s
  Total training time: 0.75h
```

### Example 2: Basic Training Plot

```bash
python plot_training_history.py checkpoints/model/history.json
```

**Creates:**
- `training_history_model.png` - Full training curves

### Example 3: Comparison with Smoothing

```bash
python plot_training_history.py \
    dcmnet/history.json noneq/history.json \
    --compare \
    --names "DCMNet (Equivariant)" "Non-Equivariant" \
    --smoothing 0.9 \
    --dpi 200
```

**Features:**
- Smoothed curves (α=0.9)
- Raw data shown as faint background
- Both models on same plots
- Best epoch markers

### Example 4: Full Analysis with Parameters

```bash
python plot_training_history.py \
    comparisons/test1/dcmnet_equivariant/history.json \
    comparisons/test1/noneq_model/history.json \
    --compare \
    --names "DCMNet" "Non-Eq" \
    --params \
        comparisons/test1/dcmnet_equivariant/best_params.pkl \
        comparisons/test1/noneq_model/best_params.pkl \
    --analyze-params \
    --convergence \
    --smoothing 0.9 \
    --output-dir analysis_results \
    --dpi 200
```

**Creates 6 plots:**
1. `training_comparison.png` - Training curves
2. `convergence_analysis_dcmnet.png` - DCMNet convergence
3. `convergence_analysis_non-eq.png` - Non-Eq convergence
4. `parameter_analysis_dcmnet.png` - DCMNet parameters
5. `parameter_analysis_non-eq.png` - Non-Eq parameters
6. `parameter_comparison.png` - Side-by-side comparison

### Example 5: Publication Figures

```bash
python plot_training_history.py history.json \
    --params best_params.pkl \
    --analyze-params \
    --convergence \
    --format pdf \
    --dpi 300 \
    --output-dir publication
```

---

## 🎨 Visual Features

### Hatching Patterns

All plots use distinctive hatching:
- **DCMNet:** `///` (forward diagonal)
- **Non-Eq:** `\\\` (backward diagonal)
- **Different metrics:** `xxx`, `...`, `|||`

Works perfectly in grayscale!

### Color Scheme

- **DCMNet:** Blue `#2E86AB`
- **Non-Eq:** Purple `#A23B72`
- **Energy:** Green `#06A77D`
- **Forces:** Red `#F45B69`
- **Dipole:** Teal `#4ECDC4`
- **ESP:** Coral `#FF6B6B`
- **Best markers:** Gold

### Text Sizing

- Title: **16pt bold**
- Axis labels: **12-13pt bold**
- Tick labels: **10-11pt**
- Legends: **10-12pt**

### Smoothing

Exponential moving average (EMA) smoothing:
- `--smoothing 0.0` - No smoothing (raw data)
- `--smoothing 0.5` - Light smoothing
- `--smoothing 0.9` - Heavy smoothing (recommended)
- `--smoothing 0.95` - Very heavy smoothing

**Tip:** Use 0.9 for noisy training curves!

---

## 📐 Plot Details

### Training Comparison Plot

**Layout:** 4 rows × 2 columns

**Row 1 (full width):**
- Validation loss comparison
- Both models overlaid
- Gold line at best epoch
- Log scale

**Rows 2-3:**
- Energy, Forces, Dipole, ESP MAE
- Separate subplots for each metric
- Best values marked with star
- Log scale for clear visibility

**Row 4 (full width):**
- Training speed (time per epoch)
- Shows average in legend
- Useful for efficiency comparison

### Convergence Analysis Plot

**Layout:** 2 rows × 2 columns

**Plots:**
1. **Per-Epoch Improvement** - Shows when model is improving vs degrading
2. **50-Epoch Rolling Average** - Detects convergence point
3. **Training Progress** - % to best model (0-100%)
4. **Final Metrics** - Bar chart of validation MAE

**Features:**
- Green/red shading for improvement/degradation
- Convergence point detection (when improvement < 1% of max)
- 90% progress marker
- Multiple hatching patterns

### Parameter Analysis Plot

**Layout:** 2 rows × 2 columns

**Plots:**
1. **Module Parameters (Pie)** - Percentage breakdown
2. **Module Parameters (Bar)** - Absolute counts
3. **Layer Size Distribution** - Histogram (log scale)
4. **Summary Text Box** - Detailed statistics

**Information Shown:**
- Total parameter count
- Number of layers
- Module breakdown with percentages
- Largest layers
- Size statistics

### Parameter Comparison Plot

**Layout:** 1 row × 2 columns

**Plots:**
1. **Total Size Comparison** - Bar chart with values
2. **Module-Level Comparison** - Grouped bars

**Features:**
- Shows absolute difference
- Calculates percentage smaller
- Module-by-module breakdown
- Clear hatching patterns

---

## 💡 Tips & Tricks

### Tip 1: Always Use Smoothing for Long Runs

```bash
# For 100+ epochs, use heavy smoothing
python plot_training_history.py history.json --smoothing 0.9
```

### Tip 2: Check Summary First

```bash
# Quick check without plots
python plot_training_history.py history.json --summary-only

# Then generate plots if interesting
python plot_training_history.py history.json
```

### Tip 3: Batch Process Multiple Runs

```bash
# Compare all runs in a directory
for dir in checkpoints/*/; do
    python plot_training_history.py "$dir/history.json" \
        --output-dir "$dir/plots"
done
```

### Tip 4: Find Best Hyperparameters

```bash
# After hyperparameter sweep
python plot_training_history.py \
    checkpoints/hparam_*/history.json \
    --summary-only | grep "Best Val Loss"
```

### Tip 5: Publication-Ready Figures

```bash
python plot_training_history.py history.json \
    --params best_params.pkl \
    --analyze-params \
    --convergence \
    --format pdf \
    --dpi 300 \
    --smoothing 0.95
```

---

## 🔄 Workflow Integration

### After Training on HPC

```bash
# 1. Download results from Scicore
scp -r scicore:/path/to/checkpoints/my_model/ ./

# 2. Quick summary
python plot_training_history.py my_model/history.json --summary-only

# 3. Generate plots
python plot_training_history.py my_model/history.json \
    --params my_model/best_params.pkl \
    --analyze-params --convergence
```

### After Model Comparison

```bash
# Full comparison
python plot_training_history.py \
    comparisons/test1/dcmnet_equivariant/history.json \
    comparisons/test1/noneq_model/history.json \
    --compare \
    --names "DCMNet" "Non-Eq" \
    --params \
        comparisons/test1/dcmnet_equivariant/best_params.pkl \
        comparisons/test1/noneq_model/best_params.pkl \
    --analyze-params \
    --convergence \
    --smoothing 0.9 \
    --output-dir paper_figures \
    --dpi 300 \
    --format pdf
```

### Hyperparameter Analysis

```bash
# Compare different learning rates
python plot_training_history.py \
    hparam_lr0.001/history.json \
    hparam_lr0.0005/history.json \
    --compare \
    --names "LR=0.001" "LR=0.0005" \
    --smoothing 0.9
```

---

## 📊 Understanding the Plots

### Training Curves

**What to look for:**
- ✅ Smooth decrease in validation loss
- ✅ No divergence between train/val
- ✅ Stable convergence
- ⚠️ Oscillations = learning rate too high
- ⚠️ Flat early = learning rate too low
- ⚠️ Train << Val = overfitting

### Convergence Analysis

**Improvement Rate:**
- Positive values = model improving
- Negative values = model degrading
- Should approach zero as training progresses

**Rolling Average:**
- Detects when learning plateaus
- Convergence point = when improvement < 1% of max

**Progress:**
- Shows % of total improvement achieved
- 90% marker helps estimate remaining time

### Parameter Analysis

**Module Pie Chart:**
- Shows which modules dominate model size
- Useful for understanding architecture

**Layer Size Distribution:**
- Histogram on log scale
- Shows if parameters are balanced
- Peaks indicate common layer sizes

**Summary Box:**
- Total parameters and breakdown
- Largest layers identified
- Quick reference for model size

### Parameter Comparison

**Total Size:**
- Direct comparison of model complexity
- Shows which model is more parameter-efficient

**Module Breakdown:**
- Identifies where models differ
- Helps understand architectural choices
- E.g., DCMNet has larger message passing, Non-Eq has simpler MLPs

---

## 🎓 Advanced Usage

### Custom Smoothing

```bash
# Light smoothing (preserve detail)
--smoothing 0.5

# Heavy smoothing (reduce noise)
--smoothing 0.9

# Very heavy (presentations)
--smoothing 0.95
```

### Multiple Formats

```bash
# Create both screen and print versions
python plot_training_history.py history.json --dpi 150 --format png
python plot_training_history.py history.json --dpi 300 --format pdf
```

### Selective Plotting

```bash
# Just training curves
python plot_training_history.py history.json

# Just convergence
python plot_training_history.py history.json --convergence

# Just parameters
python plot_training_history.py history.json \
    --params best_params.pkl --analyze-params
```

---

## 🔍 What Each Metric Means

### Loss Curves

- **Train Loss:** Model performance on training data
- **Val Loss:** Performance on validation (unseen) data
- **Gap:** Large gap suggests overfitting

### MAE Metrics

- **Energy MAE:** Energy prediction error
- **Forces MAE:** Force prediction error
- **Dipole MAE:** Dipole moment error
- **ESP MAE:** Electrostatic potential error

**Lower is better** for all!

### Training Speed

- **Epoch Time:** How long each epoch takes
- **Faster model:** Completes epochs quicker
- **Note:** DCMNet usually slower (more complex)

### Parameter Counts

- **Total:** Overall model size
- **By Module:** Which parts are largest
- **Layer Distribution:** How parameters are allocated

---

## 📁 File Structure

### Input Files

**Required:**
- `history.json` - Training curves and metrics

**Optional:**
- `best_params.pkl` - Model parameters (for analysis)

### Output Files

**Training Plots:**
- `training_history_<model>.png` - Single model
- `training_comparison.png` - Two models

**Convergence:**
- `convergence_analysis_<model>.png`

**Parameters:**
- `parameter_analysis_<model>.png` - Individual
- `parameter_comparison.png` - Comparison

---

## 🐛 Troubleshooting

### Problem: "JAX not available"

Parameter analysis requires JAX. If not available, you'll get limited analysis.

**Solution:**
```bash
pip install jax jaxlib
# or
conda install jax
```

### Problem: Plots look noisy

**Solution:** Use smoothing:
```bash
--smoothing 0.9
```

### Problem: Can't read parameters

**Solution:** Ensure pickle file is from Flax/JAX checkpoint:
```bash
python -c "import pickle; print(pickle.load(open('best_params.pkl', 'rb')).keys())"
```

### Problem: Text too small

**Solution:** Increase DPI or edit font sizes in script.

---

## 🎯 Common Scenarios

### Scenario 1: After Training

```bash
# Check if training succeeded
python plot_training_history.py checkpoints/run/history.json --summary-only

# Visualize
python plot_training_history.py checkpoints/run/history.json \
    --convergence --dpi 150
```

### Scenario 2: Compare Hyperparameters

```bash
python plot_training_history.py \
    run_lr0.001/history.json run_lr0.0005/history.json \
    --compare --names "LR=0.001" "LR=0.0005" \
    --smoothing 0.9
```

### Scenario 3: Model Selection

```bash
# Compare architectures
python plot_training_history.py \
    dcmnet/history.json noneq/history.json \
    --compare --names "DCMNet" "Non-Eq" \
    --params dcmnet/best_params.pkl noneq/best_params.pkl \
    --analyze-params
```

### Scenario 4: Publication Figures

```bash
python plot_training_history.py hist1.json hist2.json \
    --compare --names "Model A" "Model B" \
    --smoothing 0.95 \
    --format pdf --dpi 300 \
    --output-dir paper_figures
```

---

## ✨ Key Features

✅ **Comprehensive** - Training, convergence, parameters, all in one tool  
✅ **Smart smoothing** - Reduces noise while showing raw data  
✅ **Comparison mode** - Side-by-side analysis  
✅ **Parameter trees** - Understand model architecture  
✅ **Hatching patterns** - Grayscale/print friendly  
✅ **Large text** - Presentation ready  
✅ **Multiple formats** - PNG, PDF, SVG, JPG  
✅ **Text summaries** - Quick checks without plotting  
✅ **Convergence detection** - Automated analysis  

---

## 📚 Related Tools

- `plot_comparison_results.py` - For comparison result visualization
- `compare_models.py` - For running model comparisons
- `trainer.py` - For training models

---

## 🎨 Example Output

The tool generates beautiful, publication-ready plots with:
- Smooth curves overlaid on raw data
- Best epoch markers (gold stars)
- Color-coded metrics
- Hatching for accessibility
- Clear legends and annotations
- Professional typography

**Perfect for:**
- Progress monitoring
- Model comparison
- Hyperparameter tuning
- Publications and presentations
- Understanding model architecture

---

**Happy analyzing! 📊✨**

