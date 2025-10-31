# CO2 Example - Data Preparation and Visualization

This directory contains scripts for preparing CO2 molecular data for DCMnet/PhysnetJax training.

## ⚠️ Training Data Not Included

The actual training data files are **not committed to Git** due to their large size (>300 MB). 

### To Generate Training Data:

1. **Obtain source data**:
   - Place `energies_forces_dipoles.npz` and `grids_esp.npz` in `/home/ericb/testdata/`
   - Or modify paths in the scripts

2. **Run the data preparation script**:
   ```bash
   python fix_and_split_data.py
   ```

3. **Output will be created in**:
   - `training_data_fixed/` (ASE-standard units, 8:1:1 split)

## Scripts

### 1. `co2data.py` - Data Visualization
Generates 7 publication-quality plots:
- Energy, force, and dipole distributions
- ESP 2D slices (XY, XZ, YZ planes)
- ESP 3D scatter with molecular overlay

```bash
python co2data.py
# Output: plots/*.png
```

### 2. `prepare_training_data.py` - Unit Validation
Validates units and creates basic 8:1:1 splits (diagnostic only).

```bash
python prepare_training_data.py
# Output: training_data/
```

### 3. `fix_and_split_data.py` - Main Data Preparation
Converts to ASE-standard units and creates training splits.

```bash
python fix_and_split_data.py
# Output: training_data_fixed/
```

**This is the main script to use!**

### 4. `merge_charge_files.py` - Merge Multiwfn Charges
Combines multiple Multiwfn charge analysis NPZ files.

```bash
python merge_charge_files.py /path/to/charge/files/ \
  -o co2_charges.npz \
  --filter-method mp2 \
  --compress
```

## Output Data Format

### Training Data (ASE Standard Units)

**energies_forces_dipoles_{train,valid,test}.npz**:
- `R`: (n, 60, 3) Angstrom
- `Z`: (n, 60) atomic numbers
- `N`: (n,) number of atoms
- `E`: (n,) eV
- `F`: (n, 60, 3) eV/Angstrom
- `Dxyz`: (n, 3) Debye

**grids_esp_{train,valid,test}.npz**:
- `R`, `Z`, `N`: Same as above
- `esp`: (n, 3000) Hartree/e
- `vdw_surface`: (n, 3000, 3) Angstrom
- `Dxyz`: (n, 3) Debye
- Grid metadata: `grid_dims`, `grid_origin`, `grid_axes`

### Splits (Reproducible, seed=42)
- Train: 8,000 samples (80%)
- Valid: 1,000 samples (10%)
- Test: 1,000 samples (10%)

## Unit Conversions Applied

| Property | Original | Final (ASE) | Factor |
|----------|----------|-------------|--------|
| R | Angstrom | Angstrom | - |
| E | Hartree | eV | ×27.211386 |
| F | Hartree/Bohr | eV/Angstrom | ×51.42208 |
| vdw_surface | Grid indices | Angstrom | Grid spacing |

## Documentation

- `DATA_PREPARATION_SUMMARY.md` - Complete pipeline overview
- `COMPLETE_SUMMARY.md` - Session summary
- `MERGE_CHARGES_USAGE.md` - Charge merger instructions
- `training_data_fixed/README.md` - Training data description (generated)
- `training_data_fixed/UNITS_REFERENCE.txt` - Quick reference (generated)

## Visualizations

All plots are saved to `plots/` directory:
- `energy_distribution.png`
- `force_distributions.png`
- `dipole_distribution.png`
- `esp_distribution.png`
- `esp_slices.png`
- `esp_3d_scatter.png`
- `esp_3d_with_molecule.png`

## Notes

1. **Training data files are too large for Git** - generate locally using the scripts
2. **Plots are excluded from Git** (`.gitignore` includes `*.png`)
3. **All units are ASE-standard** - compatible with ASE, SchNetPack, etc.
4. **Data quality validated** - strict unit checking performed

## Quick Start

```bash
# 1. Generate training data
python fix_and_split_data.py

# 2. Create visualizations
python co2data.py

# 3. (Optional) Merge charge files
python merge_charge_files.py /path/to/charges/ -o charges.npz

# 4. Start training
python -m mmml.dcmnet.train_runner --data-dir training_data_fixed/
```

## Questions?

See the detailed documentation files for more information.

