# Comparison Results Plotting CLI

A comprehensive command-line tool for visualizing model comparison results.

## 🚀 Quick Start

```bash
# Plot everything from a comparison
python plot_comparison_results.py comparisons/my_test/comparison_results.json

# Just show text summary
python plot_comparison_results.py comparisons/my_test/comparison_results.json --summary-only
```

## 📊 Features

### 1. Multiple Plot Types

- **Performance**: MAE for energy, forces, dipole, ESP
- **Efficiency**: Training time, inference time, parameters
- **Equivariance**: Rotation and translation error tests
- **Overview**: Combined plot with all metrics
- **Multiple Comparisons**: Compare across different runs

### 2. Customization Options

- Output format (PNG, PDF, SVG, JPG)
- DPI settings
- Custom colors
- Figure sizes
- Show/hide values on bars

### 3. Multiple Run Comparison

Compare results across different hyperparameter settings or datasets.

---

## 📖 Usage Examples

### Basic Usage

```bash
# Plot all metrics
python plot_comparison_results.py comparisons/test1/comparison_results.json
```

**Output:**
- `performance_comparison.png`
- `efficiency_comparison.png`
- `equivariance_comparison.png`
- `overview_combined.png`

### Plot Specific Type

```bash
# Only performance metrics
python plot_comparison_results.py results.json --plot-type performance

# Only equivariance tests
python plot_comparison_results.py results.json --plot-type equivariance

# Only overview
python plot_comparison_results.py results.json --plot-type overview
```

### Customize Output

```bash
# High-resolution PDF
python plot_comparison_results.py results.json \
    --output-dir publication_figures \
    --dpi 300 \
    --format pdf

# Custom figure size
python plot_comparison_results.py results.json \
    --figsize 16,12 \
    --dpi 200

# Custom colors (hex codes)
python plot_comparison_results.py results.json \
    --colors "#FF6B6B,#4ECDC4"
```

### Compare Multiple Runs

```bash
# Compare different hyperparameter settings
python plot_comparison_results.py \
    comparisons/run1/comparison_results.json \
    comparisons/run2/comparison_results.json \
    comparisons/run3/comparison_results.json \
    --compare-multiple \
    --metric dipole_mae

# Compare different metrics
python plot_comparison_results.py run*/comparison_results.json \
    --compare-multiple \
    --metric esp_mae \
    --output-dir meta_analysis
```

### Text Summary Only

```bash
# Just print statistics, no plots
python plot_comparison_results.py results.json --summary-only
```

**Example output:**
```
======================================================================
COMPARISON RESULTS SUMMARY
======================================================================

📊 PERFORMANCE (Validation)
----------------------------------------------------------------------
  Energy MAE     : DCMNet=0.000123 | Non-Eq=0.000145 eV       | Winner: DCMNet ✅ (15.2%)
  Forces MAE     : DCMNet=0.012345 | Non-Eq=0.013456 eV/Å     | Winner: DCMNet ✅ (8.3%)
  Dipole MAE     : DCMNet=0.000456 | Non-Eq=0.000789 e·Å      | Winner: DCMNet ✅ (42.2%)
  ESP MAE        : DCMNet=0.001234 | Non-Eq=0.001567 Ha/e     | Winner: DCMNet ✅ (21.2%)

⚡ EFFICIENCY
----------------------------------------------------------------------
  Training Time  : DCMNet=2.45h | Non-Eq=1.87h
  Inference Time : DCMNet=12.34ms | Non-Eq=8.91ms
  Parameters     : DCMNet=1.23M | Non-Eq=0.98M

🔄 EQUIVARIANCE TESTS
----------------------------------------------------------------------
  Rotation Error : DCMNet=2.34e-06 ± 1.23e-06 e·Å
                   Non-Eq=1.45e-01 ± 3.21e-02 e·Å
                   ✅ DCMNet is equivariant
                   ⚠️ Non-Eq is not equivariant (expected)

  Translation Error: DCMNet=3.45e-06 ± 1.67e-06 e·Å
                     Non-Eq=4.12e-06 ± 1.89e-06 e·Å
                     ✅ Both models are translation invariant
```

---

## 🎨 Plot Descriptions

### Performance Comparison

<img src="performance_comparison.png" width="600"/>

**Shows:**
- Energy, Forces, Dipole, ESP MAE
- Side-by-side bars for each model
- Golden border on winner
- Percentage improvement annotation

**Best for:** Understanding prediction accuracy

### Efficiency Comparison

<img src="efficiency_comparison.png" width="600"/>

**Shows:**
- Training time (hours)
- Inference time (milliseconds)
- Parameter count (millions)

**Best for:** Understanding computational costs

### Equivariance Comparison

<img src="equivariance_comparison.png" width="600"/>

**Shows:**
- Rotation error (log scale)
- Translation error (log scale)
- Error bars (standard deviation)
- Reference line for "perfect" equivariance

**Best for:** Verifying symmetry properties

### Combined Overview

<img src="overview_combined.png" width="800"/>

**Shows:**
- All metrics in one figure
- Summary statistics text box
- Compact layout for presentations

**Best for:** Quick overview or slides

---

## ⚙️ Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `json_files` | One or more comparison results JSON files |

### Optional Arguments

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--plot-type` | `all`, `performance`, `efficiency`, `equivariance`, `overview` | `all` | Type of plot to generate |
| `--output-dir` | path | same as JSON | Output directory for plots |
| `--dpi` | integer | `150` | Resolution (DPI) |
| `--format` | `png`, `pdf`, `svg`, `jpg` | `png` | Output format |
| `--figsize` | `width,height` | auto | Figure size in inches |
| `--colors` | `color1,color2` | auto | Custom colors (hex codes) |
| `--no-values` | flag | False | Don't show values on bars |
| `--compare-multiple` | flag | False | Compare multiple runs |
| `--metric` | metric name | `dipole_mae` | Metric for multi-run comparison |
| `--summary-only` | flag | False | Only print text summary |

---

## 🔄 Workflow Integration

### After Training

```bash
# Run comparison
sbatch sbatch/04_compare_models.sbatch

# Wait for completion, then plot
python plot_comparison_results.py \
    comparisons/model_comparison_*/comparison_results.json
```

### For Publications

```bash
# High-quality PDFs for papers
python plot_comparison_results.py results.json \
    --output-dir paper_figures \
    --format pdf \
    --dpi 300 \
    --figsize 8,6

# Individual plots for different sections
python plot_comparison_results.py results.json \
    --plot-type performance \
    --format pdf --dpi 300

python plot_comparison_results.py results.json \
    --plot-type equivariance \
    --format pdf --dpi 300
```

### For Presentations

```bash
# Overview slide
python plot_comparison_results.py results.json \
    --plot-type overview \
    --format png \
    --dpi 200 \
    --figsize 16,10

# Or high-contrast colors
python plot_comparison_results.py results.json \
    --colors "#FF6B6B,#4ECDC4" \
    --dpi 200
```

### Hyperparameter Analysis

```bash
# After running hyperparameter sweep
python plot_comparison_results.py \
    comparisons/hparam_*/comparison_results.json \
    --compare-multiple \
    --metric dipole_mae \
    --output-dir hparam_analysis
```

---

## 📁 File Structure

### Input (JSON format)

The tool expects JSON files with this structure:

```json
{
  "dcmnet_metrics": {
    "validation": {
      "energy_mae": 0.000123,
      "forces_mae": 0.012345,
      "dipole_mae": 0.000456,
      "esp_mae": 0.001234
    },
    "training_time_hours": 2.45,
    "inference_time_ms": 12.34,
    "parameters": 1234567,
    "equivariance": {
      "rotation_error_mean": 2.34e-06,
      "rotation_error_std": 1.23e-06,
      "translation_error_mean": 3.45e-06,
      "translation_error_std": 1.67e-06
    }
  },
  "noneq_metrics": { ... }
}
```

### Output Files

Depending on `--plot-type`:

- `performance_comparison.{format}`
- `efficiency_comparison.{format}`
- `equivariance_comparison.{format}`
- `overview_combined.{format}`
- `multiple_comparisons_{metric}.{format}` (for `--compare-multiple`)

---

## 💡 Tips & Tricks

### 1. Quick Check After Training

```bash
# Just see the numbers
python plot_comparison_results.py results.json --summary-only
```

### 2. Batch Processing

```bash
# Plot all comparisons at once
for dir in comparisons/*/; do
    python plot_comparison_results.py "$dir/comparison_results.json" \
        --output-dir "$dir/plots"
done
```

### 3. Custom Color Schemes

Popular color combinations:

```bash
# Blue vs Red
--colors "#2E86AB,#A23B72"

# Green vs Orange
--colors "#06A77D,#F45B69"

# Teal vs Coral
--colors "#4ECDC4,#FF6B6B"

# Professional grayscale
--colors "#2C3E50,#95A5A6"
```

### 4. Publication-Ready Figures

```bash
# Nature/Science style (typically 89mm single column)
python plot_comparison_results.py results.json \
    --format pdf --dpi 600 --figsize 3.5,3

# Full page figure
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --figsize 8,10
```

### 5. Automated Reporting

Create a script:

```bash
#!/bin/bash
# analyze_comparison.sh

RESULT_JSON=$1

echo "Analyzing: $RESULT_JSON"

# Print summary
python plot_comparison_results.py "$RESULT_JSON" --summary-only > summary.txt

# Create all plots
python plot_comparison_results.py "$RESULT_JSON" \
    --output-dir plots \
    --format png --dpi 150

# Create publication version
python plot_comparison_results.py "$RESULT_JSON" \
    --output-dir publication \
    --format pdf --dpi 300

echo "Done! See plots/ and publication/"
```

---

## 🐛 Troubleshooting

### Problem: "matplotlib not found"

**Solution:**
```bash
pip install matplotlib
# or
conda install matplotlib
```

### Problem: "Invalid JSON"

**Solution:** Check that the JSON file is from `compare_models.py` and is valid:
```bash
python -m json.tool comparison_results.json
```

### Problem: Plots look too small/large

**Solution:** Adjust figsize:
```bash
# Larger
--figsize 16,12

# Smaller
--figsize 8,6
```

### Problem: Text overlapping in plots

**Solution:** Increase figure size or DPI:
```bash
--figsize 14,10 --dpi 200
```

---

## 📚 Related Documentation

- `COMPARISON_GUIDE.md` - Running model comparisons
- `README.md` - Project overview
- `QUICK_REFERENCE.md` - Training commands

---

## ✨ Advanced: Programmatic Use

You can also import and use the plotting functions in your own scripts:

```python
from plot_comparison_results import (
    load_comparison_results,
    plot_performance_comparison,
    plot_equivariance_comparison,
    print_summary
)

# Load data
data = load_comparison_results(Path('results.json'))

# Print summary
print_summary(data)

# Create custom plots
plot_performance_comparison(
    data, 
    output_dir=Path('my_plots'),
    dpi=300,
    colors={'dcmnet': '#FF0000', 'noneq': '#0000FF'}
)
```

---

## 🎓 Examples by Use Case

### Use Case 1: Quick Sanity Check

```bash
python plot_comparison_results.py results.json --summary-only
```

### Use Case 2: Presentation Slides

```bash
python plot_comparison_results.py results.json \
    --plot-type overview \
    --format png --dpi 150 \
    --figsize 16,9
```

### Use Case 3: Paper Figures

```bash
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 \
    --output-dir paper_figs
```

### Use Case 4: Hyperparameter Study

```bash
python plot_comparison_results.py lr_*/comparison_results.json \
    --compare-multiple \
    --metric dipole_mae \
    --output-dir hyperparam_analysis
```

### Use Case 5: Batch Processing

```bash
find comparisons/ -name "comparison_results.json" -exec \
    python plot_comparison_results.py {} --output-dir {}/plots \;
```

---

**Enjoy your visualizations! 🎨📊**

