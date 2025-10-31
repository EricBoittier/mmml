# CO2 Training Data (Unit-Corrected)

This directory contains CO2 molecular data with **corrected units** ready for DCMnet/PhysnetJax training.

## Data Corrections Applied

### 1. Atomic Coordinates (R)
- **Original**: Already in Angstroms (varying C-O bonds: 1.0-1.5 Å)
- **Fixed**: No conversion needed - already correct!
- **Note**: Dataset contains varying CO2 geometries (R1, R2, angle scans)

### 2. ESP Grid Coordinates (vdw_surface)
- **Original**: Grid index space (0-49 with unit axes)
- **Fixed**: Physical Angstroms
- **Conversion**: Applied proper grid spacing (0.25 Bohr = 0.132294 Å)

## Data Splits

- **Train**: 8000 samples (80%)
- **Valid**: 1000 samples (10%)
- **Test**: 1000 samples (10%)
- **Seed**: 42 (reproducible)

## Files

### Energy, Forces, and Dipoles
- `energies_forces_dipoles_train.npz`
- `energies_forces_dipoles_valid.npz`
- `energies_forces_dipoles_test.npz`

Each contains:
- `R`: Atomic coordinates (n_samples, 60, 3) [Angstrom]
- `Z`: Atomic numbers (n_samples, 60) [int]
- `N`: Number of atoms (n_samples,) [int]
- `E`: Energies (n_samples,) [Hartree]
- `F`: Forces (n_samples, 60, 3) [Hartree/Bohr]
- `Dxyz`: Dipole moments (n_samples, 3) [Debye]

### ESP Grids
- `grids_esp_train.npz`
- `grids_esp_valid.npz`
- `grids_esp_test.npz`

Each contains:
- `R`: Atomic coordinates (n_samples, 60, 3) [Angstrom] ← FIXED
- `Z`: Atomic numbers (n_samples, 60) [int]
- `N`: Number of atoms (n_samples,) [int]
- `esp`: ESP values (n_samples, 3000) [Hartree/e]
- `vdw_surface`: Grid coordinates (n_samples, 3000, 3) [Angstrom] ← FIXED
- `vdw_grid`: Same as vdw_surface (backward compatibility)
- `grid_dims`: Original cube dimensions (n_samples, 3)
- `grid_origin`: Original cube origins (n_samples, 3) [Bohr]
- `grid_axes`: Original cube axes (n_samples, 3, 3)
- `Dxyz`: Dipole moments (n_samples, 3) [Debye]

## Units Summary (MMML Standard)

| Property | Unit | Status |
|----------|------|--------|
| R (coordinates) | Angstrom | ✓ Fixed |
| E (energy) | Hartree | ✓ Correct |
| F (forces) | Hartree/Bohr | ✓ Correct |
| Dxyz (dipoles) | Debye | ✓ Correct |
| esp (values) | Hartree/e | ✓ Correct |
| vdw_surface | Angstrom | ✓ Fixed |

## Usage with DCMnet

```python
import numpy as np

# Load training data
train_props = np.load('training_data_fixed/energies_forces_dipoles_train.npz')
train_grids = np.load('training_data_fixed/grids_esp_train.npz')

# All units are now correct - ready to use!
R = train_props['R']  # Angstroms
E = train_props['E']  # Hartree
F = train_props['F']  # Hartree/Bohr
esp = train_grids['esp']  # Hartree/e
vdw_surface = train_grids['vdw_surface']  # Angstroms
```

## Important Notes

1. **Coordinates are now in physical Angstroms** - no scaling needed
2. **ESP grid is in physical Angstroms** - matches atomic coordinates
3. **Forces remain in Hartree/Bohr** - standard quantum chemistry unit
4. **ESP values are Hartree/e** - electrostatic potential per unit charge
5. **Splits are reproducible** - same seed (42) ensures consistency

## Validation

Run `prepare_training_data.py` to see detailed validation of the original data,
or check the `VALIDATION_REPORT.txt` for the full analysis.

Generated: fix_and_split_data.py
