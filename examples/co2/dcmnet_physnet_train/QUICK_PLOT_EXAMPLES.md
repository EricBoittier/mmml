# 🎨 Quick Plotting Examples

All plots now have **automatic enhancements**: error bars, hatching, larger text!

## 🚀 Most Common Use Cases

### 1. Quick Check
```bash
python plot_comparison_results.py results.json --summary-only
```

### 2. Generate All Plots (Enhanced!)
```bash
python plot_comparison_results.py results.json
```
**Gets you:**
- ✅ Error bars (RMSE-based)
- ✅ Cool hatching patterns (/// vs \\\)
- ✅ Larger text (publication-ready)
- ✅ Thinner, better proportions

### 3. Publication Figures
```bash
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --output-dir paper_figures
```

### 4. Compare Multiple Runs
```bash
python plot_comparison_results.py \
    run1/results.json run2/results.json run3/results.json \
    --compare-multiple --metric dipole_mae
```

## 🎯 What's Different?

### Before
```bash
# Old plots: thin lines, small text, no hatching
python plot_comparison_results.py results.json
```
→ 80-160KB files

### Now (Automatic!)
```bash
# Same command, enhanced output!
python plot_comparison_results.py results.json
```
→ 168-352KB files (more detail, better quality)

**You get automatically:**
- 📊 Error bars on all plots
- 🎯 Hatching patterns (DCMNet: ///, Non-Eq: \\\)
- 📝 Larger text (12-16pt, all bold)
- 📐 Better proportions (thinner plots)
- 🎨 Thicker lines (2.5pt)
- ✨ Publication-ready

## 📊 Visual Features

### Hatching Patterns
```
DCMNet (Equivariant):  ///   Forward diagonal
Non-Equivariant:       \\\   Backward diagonal
```

**Why hatching?**
- ✅ Works in grayscale
- ✅ Print-friendly  
- ✅ Accessible
- ✅ Professional

### Error Bars
- Based on equivariance RMSE
- DCMNet: Small errors (more precise)
- Non-Eq: Larger errors (less precise)
- Thick bars (2.5pt) with large caps

### Text Sizes
- **Title:** 16pt bold
- **Axis labels:** 13pt bold
- **Tick labels:** 11-12pt bold
- **Values:** 12pt bold

## 🎨 Customization

### Custom Colors (Hatching Auto-Added!)
```bash
python plot_comparison_results.py results.json \
    --colors "#FF6B6B,#4ECDC4"
```

### High Resolution
```bash
python plot_comparison_results.py results.json --dpi 300
```

### Specific Plot Type
```bash
# Just performance
python plot_comparison_results.py results.json --plot-type performance

# Just equivariance  
python plot_comparison_results.py results.json --plot-type equivariance

# Just overview
python plot_comparison_results.py results.json --plot-type overview
```

## 📁 Batch Processing

```bash
# Plot all comparisons
for json in comparisons/*/comparison_results.json; do
    python plot_comparison_results.py "$json"
done
```

## 💡 Pro Tips

### Tip 1: Always check summary first
```bash
python plot_comparison_results.py results.json --summary-only
# Then generate plots if needed
python plot_comparison_results.py results.json
```

### Tip 2: Create multiple versions
```bash
# Screen version
python plot_comparison_results.py results.json --dpi 150

# Print version
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --output-dir print_version
```

### Tip 3: Works with old data!
```bash
# Backward compatible - works with old JSON format
python plot_comparison_results.py old_comparison/results.json
# Automatically converts and enhances!
```

## 🎉 That's It!

**All enhancements are automatic.** Just use the tool normally and get beautiful plots! 

See `PLOTTING_ENHANCEMENTS.md` for full details.
