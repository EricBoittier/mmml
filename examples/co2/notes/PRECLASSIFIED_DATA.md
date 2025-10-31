# Preclassified Data Directory

This directory is created by `fix_and_split_cli.py` and contains processed molecular data ready for training.

## What's Inside

- **Training data** (80%): `energies_forces_dipoles_train.npz`, `grids_esp_train.npz`
- **Validation data** (10%): `energies_forces_dipoles_valid.npz`, `grids_esp_valid.npz`
- **Test data** (10%): `energies_forces_dipoles_test.npz`, `grids_esp_test.npz`
- **Split indices**: `split_indices.npz` (for reproducibility)
- **Documentation**: `README.md` (auto-generated dataset info)

## Units (ASE Standard)

All data has been converted to ASE-compatible units:

- **Coordinates**: Angstrom
- **Energies**: eV (converted from Hartree)
- **Forces**: eV/Angstrom (converted from Hartree/Bohr)
- **Dipoles**: Debye
- **ESP values**: Hartree/e
- **ESP grid**: Angstrom (converted from grid indices)

## Loading Data

```python
import numpy as np

# Load training data
train = np.load('preclassified_data/energies_forces_dipoles_train.npz')
grid = np.load('preclassified_data/grids_esp_train.npz')

# Access arrays
R = train['R']      # [Angstrom]
E = train['E']      # [eV]
F = train['F']      # [eV/Angstrom]
esp = grid['esp']   # [Hartree/e]
```

## Regenerating

To regenerate this data:

```bash
python fix_and_split_cli.py \
    --efd <path_to_efd.npz> \
    --grid <path_to_grid.npz> \
    --output-dir ./preclassified_data
```

See `CLI_USAGE.md` for detailed options.
