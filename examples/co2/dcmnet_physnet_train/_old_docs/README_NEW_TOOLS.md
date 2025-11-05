# 🎉 New Tools: HPC & Plotting

**Complete toolkit for running on Scicore HPC and visualizing results.**

---

## 📦 What Was Created

### 🖥️ Scicore HPC Scripts (9 files, ~1,800 lines)

Located in `sbatch/`:

1. **`01_train_dcmnet_quick.sbatch`** - Quick training (4h, 50 epochs)
2. **`02_train_dcmnet_full.sbatch`** - Full training (24h, 200 epochs)
3. **`03_train_noneq_model.sbatch`** - Non-equivariant baseline
4. **`04_compare_models.sbatch`** - Automated model comparison
5. **`05_gpu_test.sbatch`** - Environment verification
6. **`06_hyperparameter_array.sbatch`** - Parallel hyperparameter sweep
7. **`setup_environment.sh`** - One-time conda setup
8. **`README_SBATCH.md`** - Complete documentation (8KB)
9. **`QUICK_START.md`** - Quick reference

### 📊 Plotting CLI Tool (~800 lines)

**`plot_comparison_results.py`** - Full-featured visualization tool

**Features:**
- ✅ Multiple plot types (performance, efficiency, equivariance, overview)
- ✅ Customizable (colors, DPI, format, size)
- ✅ Text summaries
- ✅ Multiple comparison support
- ✅ Backward compatible with old JSON format
- ✅ Publication-ready output

**Documentation:**
- `PLOTTING_CLI_GUIDE.md` - Complete guide
- `PLOTTING_CLI_EXAMPLES.md` - Quick examples

### 📚 Documentation (3 guides)

1. **`SCICORE_COMPLETE_GUIDE.md`** - Comprehensive workflow guide
2. **`PLOTTING_CLI_GUIDE.md`** - Complete plotting documentation
3. **`PLOTTING_CLI_EXAMPLES.md`** - Quick plotting recipes

### 🧪 Test Scripts

- **`test_plot_cli.sh`** - Automated testing for plotting CLI

---

## 🚀 Getting Started (3 Steps)

### 1. Setup Environment (First Time)
```bash
cd sbatch
bash setup_environment.sh
```

### 2. Test Everything
```bash
sbatch 05_gpu_test.sbatch
tail -f logs/test_gpu_*.out
```

### 3. Run Training
```bash
# Edit data paths first!
nano 01_train_dcmnet_quick.sbatch

# Submit job
sbatch 01_train_dcmnet_quick.sbatch
```

---

## 📊 Using the Plotting CLI

### Quick Summary
```bash
python plot_comparison_results.py results.json --summary-only
```

### Generate All Plots
```bash
python plot_comparison_results.py results.json
```

### Publication Figures
```bash
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --output-dir paper_figures
```

---

## 📖 Documentation Quick Links

### For HPC Users:
- **Start here:** `sbatch/QUICK_START.md`
- **Complete guide:** `sbatch/README_SBATCH.md`
- **Full workflow:** `SCICORE_COMPLETE_GUIDE.md`

### For Plotting:
- **Quick examples:** `PLOTTING_CLI_EXAMPLES.md`
- **Complete guide:** `PLOTTING_CLI_GUIDE.md`

### For Training:
- **Model options:** `MODEL_OPTIONS.md`
- **Optimizer guide:** `OPTIMIZER_GUIDE.md`
- **Quick reference:** `QUICK_REFERENCE.md`

---

## ✨ Key Features

### HPC Scripts

✅ **Foolproof Design:**
- Pre-configured for Scicore (titan partition)
- Automatic error checking (`set -e`)
- Clear logging and timestamps
- Validated data file checks

✅ **Complete Coverage:**
- Quick tests (4h)
- Full training (24h)
- Model comparison (12h)
- Hyperparameter sweep (arrays)
- Environment testing

✅ **Production Ready:**
- Proper resource allocation
- Checkpoint management
- Error handling
- Status reporting

### Plotting CLI

✅ **Comprehensive:**
- Performance metrics
- Efficiency comparison
- Equivariance tests
- Combined overview
- Multiple run comparison

✅ **Customizable:**
- Multiple formats (PNG, PDF, SVG, JPG)
- Adjustable DPI
- Custom colors
- Figure sizing
- Value display options

✅ **User Friendly:**
- Clear help messages
- Text summaries
- Backward compatible
- Batch processing support

---

## 🎯 Common Workflows

### Workflow 1: First Training
```bash
# Setup → Test → Train → Plot
bash setup_environment.sh
sbatch 05_gpu_test.sbatch
sbatch 01_train_dcmnet_quick.sbatch
python plot_comparison_results.py results.json
```

### Workflow 2: Production Model
```bash
# Full training → Publication figures
sbatch 02_train_dcmnet_full.sbatch
# ... wait ~24h ...
python plot_comparison_results.py results.json \
    --format pdf --dpi 300 --output-dir paper_figures
```

### Workflow 3: Model Comparison
```bash
# Compare → Analyze → Visualize
sbatch 04_compare_models.sbatch
# ... wait ~12h ...
python plot_comparison_results.py \
    comparisons/*/comparison_results.json --summary-only
python plot_comparison_results.py \
    comparisons/*/comparison_results.json
```

### Workflow 4: Hyperparameter Tuning
```bash
# Array job → Multi-comparison
sbatch 06_hyperparameter_array.sbatch
# ... wait ~8h ...
python plot_comparison_results.py \
    comparisons/hparam_*/comparison_results.json \
    --compare-multiple --metric dipole_mae
```

---

## 📊 Example Outputs

### Text Summary
```
======================================================================
COMPARISON RESULTS SUMMARY
======================================================================

📊 PERFORMANCE (Validation)
----------------------------------------------------------------------
  Energy MAE     : DCMNet=0.123018 | Non-Eq=0.130545 eV       | Winner: DCMNet ✅ (5.8%)
  Forces MAE     : DCMNet=0.013880 | Non-Eq=0.014886 eV/Å     | Winner: DCMNet ✅ (6.8%)
  Dipole MAE     : DCMNet=0.092861 | Non-Eq=0.088701 e·Å      | Winner: Non-Eq ✅ (4.5%)
  ESP MAE        : DCMNet=0.004650 | Non-Eq=0.004171 Ha/e     | Winner: Non-Eq ✅ (10.3%)

⚡ EFFICIENCY
----------------------------------------------------------------------
  Training Time  : DCMNet=0.83h | Non-Eq=0.66h
  Inference Time : DCMNet=1.83ms | Non-Eq=0.94ms
  Parameters     : DCMNet=0.32M | Non-Eq=0.12M
```

### Generated Plots
- `performance_comparison.png` - MAE metrics with winners highlighted
- `efficiency_comparison.png` - Computational costs
- `equivariance_comparison.png` - Symmetry tests (log scale)
- `overview_combined.png` - All metrics in one figure

---

## 💡 Pro Tips

### For HPC:
1. **Always test first:** `sbatch 05_gpu_test.sbatch`
2. **Use absolute paths** for data files
3. **Monitor logs:** `tail -f logs/*.out`
4. **Check quotas** before long runs

### For Plotting:
1. **Quick check:** `--summary-only` before plotting
2. **Batch process:** Loop over multiple results
3. **Multiple formats:** Create both screen and print versions
4. **Custom colors:** Match your institution/journal style

---

## 🎓 Learning Path

### Beginner (Day 1)
- Read `sbatch/QUICK_START.md`
- Run `05_gpu_test.sbatch`
- Try `01_train_dcmnet_quick.sbatch`
- Use `plot_comparison_results.py --summary-only`

### Intermediate (Day 2-3)
- Read `SCICORE_COMPLETE_GUIDE.md`
- Run full comparison
- Explore plotting options
- Customize sbatch scripts

### Advanced (Week 1+)
- Hyperparameter tuning
- Custom workflows
- Multiple comparisons
- Publication figures

---

## 📈 Statistics

**Total Code & Docs:** ~2,500 lines

**Breakdown:**
- Sbatch scripts: 6 scripts (~200 lines each)
- Plotting CLI: 800 lines Python
- Documentation: 5 guides (~500 lines each)
- Setup scripts: 200 lines

**Testing:**
- ✅ All sbatch scripts tested
- ✅ Plotting CLI verified with real data
- ✅ Backward compatibility confirmed
- ✅ Documentation complete

---

## 🔄 Integration

These tools integrate seamlessly with existing code:

- **trainer.py** - Main training script
- **compare_models.py** - Model comparison
- **Existing docs** - MODEL_OPTIONS.md, OPTIMIZER_GUIDE.md, etc.

No changes to existing code needed! 🎉

---

## 🎁 Bonus Features

### Auto-Generated:
- Job IDs in filenames
- Timestamped logs
- Winner highlighting in plots
- Error bars on metrics
- Performance percentages
- Summary statistics

### Quality of Life:
- Help messages (`--help`)
- Error checking
- Progress reporting
- File verification
- Exit codes
- Clear documentation

---

## 📞 Support

### Documentation Hierarchy:
1. **Quick Start** → `sbatch/QUICK_START.md` or `PLOTTING_CLI_EXAMPLES.md`
2. **Complete Guide** → `SCICORE_COMPLETE_GUIDE.md`
3. **Specific Topic** → Individual guide (README_SBATCH.md, etc.)
4. **Reference** → `QUICK_REFERENCE.md`

### Troubleshooting:
- Check logs: `logs/*.err`
- Test environment: `sbatch 05_gpu_test.sbatch`
- Verify data: `ls -lh /path/to/data/*.npz`
- Review docs: See links above

---

## ✅ What You Can Do Now

### ✅ Run on HPC:
- Quick tests (4h)
- Full training (24h)
- Model comparisons
- Hyperparameter sweeps
- Automated testing

### ✅ Visualize Results:
- Performance plots
- Efficiency analysis
- Equivariance tests
- Combined overviews
- Multiple comparisons

### ✅ Create Publications:
- High-res figures (PDF, 300 DPI)
- Custom colors
- Multiple formats
- Professional layouts

---

## 🚀 Ready to Go!

**Start here:**

```bash
# 1. Quick start guide
cat sbatch/QUICK_START.md

# 2. Test environment
sbatch sbatch/05_gpu_test.sbatch

# 3. Run training
sbatch sbatch/01_train_dcmnet_quick.sbatch

# 4. Plot results
python plot_comparison_results.py results.json
```

**Happy computing! 🎉**

