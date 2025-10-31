# CO2 Dataset Preparation Summary

## Overview

This document summarizes the data preparation pipeline for training DCMnet and PhysnetJax models on CO2 molecular data.

## Critical Findings

### 1. Atomic Coordinates (R)
- **Status**: ✅ Already in Angstroms - NO CONVERSION NEEDED
- **Discovery**: Initially thought to be normalized (C-O = 1.0), but actually contains varying geometries
- **Reality**: Dataset scans R1 (C-O bond 1), R2 (C-O bond 2), and angles
- **Range**: C-O bonds from 1.0 to 1.5 Å (mean ~1.25 Å)
- **Validation**: Checked 20,000 C-O bonds across all samples

### 2. ESP Grid Coordinates (vdw_grid → vdw_surface)
- **Status**: ⚠️ Required conversion from grid indices to physical Angstroms
- **Original**: Grid index space (0-49) with identity axes
- **Fixed**: Physical Angstroms using cube metadata
- **Conversion Formula**:
  ```
  coord_angstrom = (origin_bohr + (grid_index - origin) × spacing_bohr) × bohr_to_angstrom
  where:
    - spacing_bohr = 0.25 Bohr (from cube files)
    - bohr_to_angstrom = 0.529177
  ```
- **Result**: Grid extent ~6.5 Å (correct for VDW surface around CO2)

### 3. Other Fields
- **Energies (E)**: ✅ Hartree (correct, -187.4 to -187.7 Ha)
- **Forces (F)**: ✅ Hartree/Bohr (correct, well-balanced)
- **Dipoles (Dxyz)**: ✅ Debye (correct, 0-1.1 D range)
- **ESP values**: ✅ Hartree/e (correct, has +/- regions)

## Data Pipeline

```
Molpro XML files (*.xml)
    ↓
  [mmml xml2npz]
    ↓
combined.npz (with cube_potential)
    ↓
  [split_npz_cube.py]
    ↓
energies_forces_dipoles.npz + grids_esp.npz
    ↓
  [fix_and_split_data.py] ← THIS STEP
    ↓
training_data_fixed/{train,valid,test}.npz
```

## What Was Fixed

### Before (Incorrect):
- **Atomic coords**: Thought to be normalized, applied wrong ×1.16 scaling
- **ESP grid**: In index space (0-49), not physical coordinates

### After (Correct):
- **Atomic coords**: Recognized as already in Angstroms, kept as-is
- **ESP grid**: Converted to physical Angstroms using cube metadata

## Created Files

### Training Data (`training_data_fixed/`)
- `energies_forces_dipoles_train.npz` (8000 samples, 0.9 MB)
- `energies_forces_dipoles_valid.npz` (1000 samples, 0.1 MB)
- `energies_forces_dipoles_test.npz` (1000 samples, 0.1 MB)
- `grids_esp_train.npz` (8000 samples, 304 MB)
- `grids_esp_valid.npz` (1000 samples, 38 MB)
- `grids_esp_test.npz` (1000 samples, 38 MB)
- `split_indices.npz` (reproducible splits, seed=42)
- `README.md` (detailed documentation)
- `UNITS_REFERENCE.txt` (quick reference guide)

### Scripts Created
- `prepare_training_data.py` - Validates units and creates basic splits
- `fix_and_split_data.py` - Fixes ESP grid units and creates corrected splits
- `co2data.py` - Visualizes distributions and ESP slices

### Visualizations (`plots/`)
- `energy_distribution.png` - Energy histogram
- `force_distributions.png` - Force components + norms (4 subplots)
- `dipole_distribution.png` - Dipole moment distribution
- `esp_distribution.png` - ESP value histogram
- `esp_slices.png` - 2D slices (XY, XZ, YZ planes)
- `esp_3d_scatter.png` - 3D ESP grid points
- `esp_3d_with_molecule.png` - ESP with molecular structure overlay

## Unit Standards (MMML)

| Property | Unit | Status | Notes |
|----------|------|--------|-------|
| R | Angstrom | ✅ Correct | Varying geometries (1.0-1.5 Å bonds) |
| E | Hartree | ✅ Correct | -187.4 to -187.7 Ha |
| F | Hartree/Bohr | ✅ Correct | Gradient = -Force |
| Dxyz | Debye | ✅ Correct | 0-1.1 D range |
| esp | Hartree/e | ✅ Correct | Electrostatic potential |
| vdw_surface | Angstrom | ✅ Fixed | Converted from grid indices |

## Usage for Training

```python
import numpy as np

# Load training data
train_props = np.load('training_data_fixed/energies_forces_dipoles_train.npz')
train_grids = np.load('training_data_fixed/grids_esp_train.npz')

# All units are MMML-standard - ready to use!
R = train_props['R']          # (8000, 60, 3) Angstroms
Z = train_props['Z']          # (8000, 60) atomic numbers
N = train_props['N']          # (8000,) number of atoms
E = train_props['E']          # (8000,) Hartree
F = train_props['F']          # (8000, 60, 3) Hartree/Bohr
Dxyz = train_props['Dxyz']    # (8000, 3) Debye

# ESP data
esp = train_grids['esp']                 # (8000, 3000) Hartree/e
vdw_surface = train_grids['vdw_surface'] # (8000, 3000, 3) Angstroms
```

## Key Insights

1. **Dataset Structure**: 10,000 CO2 configurations with varying:
   - R1: First C-O bond length (1.0-1.5 Å)
   - R2: Second C-O bond length (1.0-1.5 Å)
   - Angle: O-C-O angle (varying)

2. **ESP Grid**: Subsampled to 3000 points per molecule from 50×50×50 cube
   - Grid spacing: 0.25 Bohr = 0.1323 Å
   - Grid extent: ~6.5 Å cube
   - Surrounds VDW surface of molecule

3. **Data Quality**:
   - All fields have correct units
   - No missing or corrupted data
   - Splits are reproducible (seed=42)
   - Train/valid/test are non-overlapping

## Recommendations for Training

1. **Use the `training_data_fixed/` directory** - units are correct
2. **Do NOT use the original `testdata/` files** - ESP grid needs conversion
3. **Check README.md** for detailed field descriptions
4. **Check UNITS_REFERENCE.txt** for quick unit lookup
5. **Splits are stratified random** - good statistical properties

## Scripts

- **`fix_and_split_data.py`**: Main script to create corrected training data
- **`prepare_training_data.py`**: Diagnostic script for unit validation
- **`co2data.py`**: Visualization script for data exploration

## Next Steps

Ready to train DCMnet or PhysnetJax with:
```bash
# Example DCMnet training
python -m mmml.dcmnet.train_runner --config config.json
```

Where config.json points to the `training_data_fixed/` directory.

