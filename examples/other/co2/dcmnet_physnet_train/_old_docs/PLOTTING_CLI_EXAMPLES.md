# Plotting CLI - Quick Examples

## 🚀 Most Common Commands

### 1. Quick Summary (No Plots)

```bash
python plot_comparison_results.py comparisons/my_test/comparison_results.json --summary-only
```

**Output:**
```
📊 PERFORMANCE (Validation)
  Energy MAE     : DCMNet=0.123018 | Non-Eq=0.130545 eV       | Winner: DCMNet ✅ (5.8%)
  Forces MAE     : DCMNet=0.013880 | Non-Eq=0.014886 eV/Å     | Winner: DCMNet ✅ (6.8%)
  ...
```

### 2. Create All Plots (Default)

```bash
python plot_comparison_results.py comparisons/my_test/comparison_results.json
```

**Creates:**
- `performance_comparison.png` - MAE metrics
- `efficiency_comparison.png` - Time & parameters
- `equivariance_comparison.png` - Rotation & translation tests
- `overview_combined.png` - All-in-one figure

### 3. High-Resolution for Publications

```bash
python plot_comparison_results.py comparisons/my_test/comparison_results.json \
    --format pdf \
    --dpi 300 \
    --output-dir paper_figures
```

### 4. Single Plot Type

```bash
# Just performance
python plot_comparison_results.py results.json --plot-type performance

# Just equivariance
python plot_comparison_results.py results.json --plot-type equivariance

# Just overview
python plot_comparison_results.py results.json --plot-type overview
```

### 5. Custom Colors

```bash
python plot_comparison_results.py results.json \
    --colors "#FF6B6B,#4ECDC4" \
    --dpi 200
```

---

## 📂 Batch Processing

### Process All Comparisons

```bash
# Find and plot all comparison results
for json in comparisons/*/comparison_results.json; do
    echo "Plotting: $json"
    python plot_comparison_results.py "$json" --dpi 150
done
```

### Create Publication Figures for All

```bash
# Create PDF versions of all comparisons
for json in comparisons/*/comparison_results.json; do
    dir=$(dirname "$json")
    python plot_comparison_results.py "$json" \
        --output-dir "$dir/publication" \
        --format pdf --dpi 300
done
```

---

## 🔄 After Running Comparisons

### Complete Workflow

```bash
# 1. Run comparison on HPC
sbatch sbatch/04_compare_models.sbatch

# 2. Wait for job to finish, then download results
# (if working remotely)

# 3. Plot everything
python plot_comparison_results.py \
    comparisons/model_comparison_*/comparison_results.json \
    --dpi 150

# 4. View results
ls -lh comparisons/model_comparison_*/
```

---

## 🎨 Different Output Formats

```bash
# PNG for viewing
python plot_comparison_results.py results.json --format png --dpi 150

# PDF for papers
python plot_comparison_results.py results.json --format pdf --dpi 300

# SVG for editing in Illustrator/Inkscape
python plot_comparison_results.py results.json --format svg

# JPG for presentations
python plot_comparison_results.py results.json --format jpg --dpi 200
```

---

## 📊 Compare Multiple Runs

```bash
# Compare 3 different hyperparameter settings
python plot_comparison_results.py \
    comparisons/lr_0.001/comparison_results.json \
    comparisons/lr_0.0005/comparison_results.json \
    comparisons/lr_0.0001/comparison_results.json \
    --compare-multiple \
    --metric dipole_mae \
    --output-dir hparam_comparison
```

---

## 💡 Pro Tips

### Tip 1: Check Before Plotting

```bash
# Quick check if results look reasonable
python plot_comparison_results.py results.json --summary-only | grep "Winner"
```

### Tip 2: Create Multiple Versions

```bash
# Screen version
python plot_comparison_results.py results.json --dpi 150

# Print version
python plot_comparison_results.py results.json \
    --output-dir print_version \
    --format pdf --dpi 300
```

### Tip 3: Automated Reporting

Create a script `make_report.sh`:

```bash
#!/bin/bash
RESULT=$1

# Create directory structure
mkdir -p report/figures report/text

# Generate summary
python plot_comparison_results.py "$RESULT" --summary-only > report/text/summary.txt

# Generate plots
python plot_comparison_results.py "$RESULT" \
    --output-dir report/figures \
    --format png --dpi 150

python plot_comparison_results.py "$RESULT" \
    --output-dir report/figures \
    --format pdf --dpi 300

echo "Report created in report/"
```

---

## 🎯 Integration with Scicore Workflow

### After HPC Job Completes

```bash
# 1. Check if job finished
squeue -u $USER

# 2. Navigate to results
cd comparisons/model_comparison_JOBID/

# 3. Quick check
python ../../plot_comparison_results.py comparison_results.json --summary-only

# 4. Generate plots
python ../../plot_comparison_results.py comparison_results.json --dpi 200

# 5. Download to local machine (if needed)
# On local machine:
# scp -r scicore:/path/to/comparisons/model_comparison_JOBID/ ./
```

---

## 📈 Example Outputs

### Performance Comparison
Shows MAE for all properties with winner highlighted in gold.

### Efficiency Comparison
Shows computational costs - training time, inference time, parameters.

### Equivariance Comparison
Shows rotation and translation errors on log scale with reference lines.

### Combined Overview
All metrics in one figure with summary statistics.

---

## 🔧 Troubleshooting

### Problem: Plots look too small

```bash
# Increase DPI and figure size
python plot_comparison_results.py results.json --figsize 16,12 --dpi 200
```

### Problem: Colors don't look good

```bash
# Try different color schemes
--colors "#FF6B6B,#4ECDC4"  # Red-Teal
--colors "#06A77D,#F45B69"  # Green-Coral
--colors "#2E86AB,#A23B72"  # Blue-Purple (default)
```

### Problem: Text overlapping

```bash
# Increase figure size
--figsize 14,10
```

---

## 📚 See Also

- `PLOTTING_CLI_GUIDE.md` - Complete documentation
- `COMPARISON_GUIDE.md` - Running comparisons
- `README.md` - Project overview

