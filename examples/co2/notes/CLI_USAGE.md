# Fix and Split Data CLI - Usage Guide

The `fix_and_split_cli.py` tool processes molecular NPZ data files, converts units to ASE standards, and creates train/validation/test splits.

## Quick Start

```bash
# Basic usage with default 8:1:1 split
python fix_and_split_cli.py \
    --efd /path/to/energies_forces_dipoles.npz \
    --grid /path/to/grids_esp.npz \
    --output-dir ./preclassified_data
```

## What It Does

The CLI performs the following operations:

1. **Validates atomic coordinates** - Checks if coordinates are in Angstroms
2. **Converts energies** - Hartree → eV (×27.211386)
3. **Converts forces** - Hartree/Bohr → eV/Angstrom (×51.42208)
4. **Fixes ESP grid coordinates** - Index space → physical Angstroms
5. **Creates data splits** - Train/Valid/Test with configurable ratios
6. **Validates output** - Ensures all conversions are correct
7. **Saves split datasets** - Creates separate NPZ files for each split
8. **Generates documentation** - Creates README.md with data details

## Command-Line Options

### Required Arguments

- `--efd PATH` or `--energies-forces-dipoles PATH`
  - Path to the energies, forces, and dipoles NPZ file
  - Must contain keys: `R`, `Z`, `N`, `E`, `F`, `Dxyz`

- `--grid PATH` or `--grids-esp PATH`
  - Path to the ESP grids NPZ file
  - Must contain keys: `R`, `Z`, `N`, `esp`, `vdw_grid`, `grid_dims`, `grid_origin`, `grid_axes`

- `--output-dir PATH` or `-o PATH`
  - Directory where processed data will be saved
  - Will be created if it doesn't exist

### Optional Arguments

- `--train-frac FLOAT` (default: 0.8)
  - Fraction of data for training set

- `--valid-frac FLOAT` (default: 0.1)
  - Fraction of data for validation set

- `--test-frac FLOAT` (default: 0.1)
  - Fraction of data for test set
  - Note: All three fractions must sum to 1.0

- `--seed INT` (default: 42)
  - Random seed for reproducible splits

- `--cube-spacing FLOAT` (default: 0.25)
  - Grid spacing in Bohr from original cube files
  - Adjust if your cube files use different spacing

- `--skip-validation`
  - Skip validation checks (faster but not recommended)

- `--quiet` or `-q`
  - Suppress detailed output (only show errors)

## Usage Examples

### Example 1: Default Configuration

Process data with default 8:1:1 split:

```bash
python fix_and_split_cli.py \
    --efd /home/user/testdata/energies_forces_dipoles.npz \
    --grid /home/user/testdata/grids_esp.npz \
    --output-dir ./preclassified_data
```

### Example 2: Custom Split Ratios

Create a 70:15:15 split for more validation/test data:

```bash
python fix_and_split_cli.py \
    --efd data.npz \
    --grid grids.npz \
    --output-dir ./training_data \
    --train-frac 0.7 \
    --valid-frac 0.15 \
    --test-frac 0.15
```

### Example 3: Different Cube Spacing

If your cube files use 0.5 Bohr spacing:

```bash
python fix_and_split_cli.py \
    --efd data.npz \
    --grid grids.npz \
    --output-dir ./training_data \
    --cube-spacing 0.5
```

### Example 4: Quiet Mode

Process without verbose output:

```bash
python fix_and_split_cli.py \
    --efd data.npz \
    --grid grids.npz \
    --output-dir ./training_data \
    --quiet
```

### Example 5: Custom Seed for Different Splits

Use a different random seed:

```bash
python fix_and_split_cli.py \
    --efd data.npz \
    --grid grids.npz \
    --output-dir ./training_data \
    --seed 12345
```

## Output Files

The CLI creates the following files in the output directory:

### Data Files

- `energies_forces_dipoles_train.npz` - Training set (EFD data)
- `energies_forces_dipoles_valid.npz` - Validation set (EFD data)
- `energies_forces_dipoles_test.npz` - Test set (EFD data)
- `grids_esp_train.npz` - Training set (ESP grid data)
- `grids_esp_valid.npz` - Validation set (ESP grid data)
- `grids_esp_test.npz` - Test set (ESP grid data)
- `split_indices.npz` - Indices used for splitting (for reproducibility)

### Documentation

- `README.md` - Detailed description of the data and corrections applied

## Data Format

### Energy/Forces/Dipoles Files

Each NPZ file contains:
- `R`: Atomic coordinates [Angstrom] - shape: (n_samples, max_atoms, 3)
- `Z`: Atomic numbers - shape: (n_samples, max_atoms)
- `N`: Number of atoms per sample - shape: (n_samples,)
- `E`: Energies [eV] - shape: (n_samples,)
- `F`: Forces [eV/Angstrom] - shape: (n_samples, max_atoms, 3)
- `Dxyz`: Dipole moments [Debye] - shape: (n_samples, 3)

### ESP Grid Files

Each NPZ file contains:
- `R`: Atomic coordinates [Angstrom] - shape: (n_samples, max_atoms, 3)
- `Z`: Atomic numbers - shape: (n_samples, max_atoms)
- `N`: Number of atoms per sample - shape: (n_samples,)
- `esp`: ESP values [Hartree/e] - shape: (n_samples, n_grid_points)
- `vdw_surface`: Grid coordinates [Angstrom] - shape: (n_samples, n_grid_points, 3)
- `vdw_grid`: Same as vdw_surface (backward compatibility)
- `grid_dims`: Original cube dimensions - shape: (n_samples, 3)
- `grid_origin`: Original cube origins [Bohr] - shape: (n_samples, 3)
- `grid_axes`: Original cube axes - shape: (n_samples, 3, 3)
- `Dxyz`: Dipole moments [Debye] - shape: (n_samples, 3)

## Unit Conversions

| Property | Original | Converted | Factor |
|----------|----------|-----------|--------|
| Coordinates (R) | Angstrom* | Angstrom | 1.0 |
| Energies (E) | Hartree | eV | 27.211386 |
| Forces (F) | Hartree/Bohr | eV/Angstrom | 51.42208 |
| Dipoles (Dxyz) | Debye | Debye | 1.0 |
| ESP values | Hartree/e | Hartree/e | 1.0 |
| ESP grid | Grid indices | Angstrom | Custom† |

*Auto-detected and converted from Bohr if needed  
†Uses origin, spacing, and axes from cube file metadata

## Loading Data in Python

```python
import numpy as np

# Load training data
train_efd = np.load('preclassified_data/energies_forces_dipoles_train.npz')
train_grid = np.load('preclassified_data/grids_esp_train.npz')

# Access data
R = train_efd['R']        # Atomic coordinates [Angstrom]
E = train_efd['E']        # Energies [eV]
F = train_efd['F']        # Forces [eV/Angstrom]
Z = train_efd['Z']        # Atomic numbers
N = train_efd['N']        # Number of atoms
Dxyz = train_efd['Dxyz']  # Dipoles [Debye]

esp = train_grid['esp']             # ESP values [Hartree/e]
vdw_surface = train_grid['vdw_surface']  # Grid coords [Angstrom]

# All units are ASE-standard!
print(f"Loaded {len(R)} training samples")
print(f"Energy range: [{E.min():.2f}, {E.max():.2f}] eV")
```

## Validation

The CLI performs automatic validation to ensure:

1. **Coordinates are in Angstroms** - Checks bond lengths for common molecules
2. **Energy conversion is correct** - Verifies reasonable eV values
3. **Force conversion is correct** - Checks force magnitudes
4. **ESP grid is in physical space** - Ensures proper spatial extent
5. **Grid surrounds molecule** - Validates spatial relationship

If validation fails, the CLI will stop and report the issue. Use `--skip-validation` to bypass checks (not recommended).

## Troubleshooting

### Error: "Split fractions must sum to 1.0"

Ensure your train/valid/test fractions add up to exactly 1.0:

```bash
# Good
--train-frac 0.7 --valid-frac 0.15 --test-frac 0.15  # = 1.0

# Bad
--train-frac 0.8 --valid-frac 0.2 --test-frac 0.1   # = 1.1
```

### Error: "File not found"

Check that your input files exist and paths are correct:

```bash
# Use absolute paths if needed
python fix_and_split_cli.py \
    --efd /absolute/path/to/energies_forces_dipoles.npz \
    --grid /absolute/path/to/grids_esp.npz \
    --output-dir ./output
```

### Warning: "Validation failed"

This usually means:
- Units are not as expected (check your input data)
- Grid spacing is incorrect (try adjusting `--cube-spacing`)
- Data format is non-standard

Review the validation output to see which checks failed.

### Output directory already exists

The CLI will overwrite existing files in the output directory. Make sure to use a unique directory name or back up existing data.

## Integration with Training Pipelines

The output data is ready to use with:
- **DCMnet** - Direct loading of NPZ files
- **PhysnetJax** - ASE-compatible units
- **SchNetPack** - Standard molecular representation
- **ASE** - All units follow ASE conventions

Example DCMnet integration:

```python
import numpy as np
from mmml.data import load_co2_data

# Load processed data
train_data = {
    'efd': np.load('preclassified_data/energies_forces_dipoles_train.npz'),
    'grid': np.load('preclassified_data/grids_esp_train.npz')
}

# Ready to use with DCMnet trainer
# All units are correct - no further conversion needed!
```

## Tips

1. **Always validate first** - Don't use `--skip-validation` unless you're certain your data is correct
2. **Check README.md** - The generated README contains detailed information about your specific dataset
3. **Use consistent seeds** - Same seed ensures reproducible splits across runs
4. **Back up raw data** - Keep original NPZ files before processing
5. **Test with small datasets first** - Verify the pipeline works before processing large datasets

## Getting Help

```bash
# Show all options
python fix_and_split_cli.py --help

# Show version and examples
python fix_and_split_cli.py -h
```

For questions or issues, refer to the main project documentation or open an issue on the repository.

