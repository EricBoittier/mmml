# CO2 Examples - Migrated to CLI

## Status

Most useful scripts from this directory have been **generalized and moved to `mmml/cli`** for reuse across all projects.

## What Was Migrated

### ✅ Now in mmml/cli

| Original Script | New CLI Tool | Purpose |
|----------------|--------------|---------|
| fix_and_split_cli.py | split_dataset.py | Train/valid/test splitting + unit conversion |
| co2data.py | explore_data.py | Dataset exploration and visualization |

### Use the CLI Tools Instead!

**Before (CO2-specific):**
```bash
cd examples/co2
python fix_and_split_cli.py --efd data.npz --grid grids.npz -o out/
python co2data.py  # Hardcoded paths
```

**After (General, works anywhere):**
```bash
# From any directory
python -m mmml.cli.split_dataset --efd data.npz --grid grids.npz -o out/ --convert-units
python -m mmml.cli.explore_data data.npz --plots --output-dir exploration
```

## Scripts Still Here (Specialized)

These remain in examples/co2 because they're specific to CO2 workflows:

- **prepare_training_data.py** - Detailed CO2 validation example
- **fix_and_split_data.py** - Alternative splitting example  
- **merge_charge_files.py** - Multiwfn charge file merging (very specific)
- **example_load_preclassified.py** - Tutorial for loading split data

## Directory Structure

```
examples/co2/
├── CO2_EXAMPLES_README.md (this file)
├── co2data.py → Use mmml.cli.explore_data instead
├── fix_and_split_cli.py → Use mmml.cli.split_dataset instead
├── prepare_training_data.py (CO2-specific example)
├── fix_and_split_data.py (alternative example)
├── merge_charge_files.py (Multiwfn-specific)
├── example_load_preclassified.py (tutorial)
└── dcmnet_physnet_train/ (specialized DCMNet training)
```

## Recommended Workflow

Instead of using these CO2-specific scripts, use the general CLI tools:

### 1. Explore Your Data
```bash
python -m mmml.cli.explore_data your_data.npz --detailed --plots --output-dir exploration
```

### 2. Clean Your Data
```bash
python -m mmml.cli.clean_data your_data.npz -o clean_data.npz --no-check-distances
```

### 3. Split for Training
```bash
# If units need conversion (Hartree → eV)
python -m mmml.cli.split_dataset clean_data.npz -o splits/ --convert-units

# If units are already correct
python -m mmml.cli.split_dataset clean_data.npz -o splits/
```

### 4. Train
```bash
python -m mmml.cli.make_training \\
    --data splits/data_train.npz \\
    --tag my_model \\
    --n_train 5000 \\
    --n_valid 1000 \\
    --num_epochs 50 \\
    --ckpt_dir checkpoints/my_model
```

### 5. Analyze
```bash
# Monitor
python -m mmml.cli.plot_training checkpoints/my_model/history.json

# Test
python -m mmml.cli.calculator --checkpoint checkpoints/my_model --test-molecule CO2

# Dynamics
python -m mmml.cli.dynamics --checkpoint checkpoints/my_model \\
    --molecule CO2 --optimize --frequencies --ir-spectra
```

## Migration Benefits

✅ **Reusable** - Works with any molecule, not just CO2  
✅ **Maintained** - Centralized in mmml/cli  
✅ **Documented** - In official CLI reference  
✅ **Tested** - Verified working  
✅ **Professional** - Production-ready code  

## Still Need CO2-Specific Scripts?

The scripts in this directory are still useful as:
- **Examples** of how to use the CLI tools
- **Reference** for CO2-specific validation
- **Templates** for creating your own specialized workflows

But for general use, prefer the CLI tools in `mmml/cli`!

---

**See also:**
- `docs/cli.rst` - Official CLI documentation
- `AI/COMPLETE_CLI_SUITE.md` - Complete tool suite overview
- `dcmnet_physnet_train/README.md` - DCMNet examples

