# Energy Preprocessing Implementation Summary

## Overview

Implemented comprehensive energy preprocessing functionality for the MMML training pipeline, including energy unit conversion, atomic energy reference subtraction, and per-atom scaling.

## Changes Made

### 1. Core Preprocessing Functions (`mmml/data/preprocessing.py`)

Added the following functions:

#### `convert_energy_units(energies, from_unit='hartree', to_unit='eV')`
- Converts energy units between eV, hartree, kcal/mol, and kJ/mol
- Uses accurate conversion factors
- Supports round-trip conversions

#### `compute_atomic_energies(energies, atomic_numbers, n_atoms, method='linear_regression')`
- Computes atomic energy references from molecular data
- Two methods:
  - `linear_regression`: Fits atomic energies using least-squares (recommended)
  - `mean`: Simple average per atom type
- Returns dictionary mapping atomic number → energy

#### `subtract_atomic_energies(energies, atomic_numbers, n_atoms, atomic_energy_refs)`
- Subtracts atomic contributions from molecular energies
- Useful for learning binding/interaction energies

#### `scale_energies_by_atoms(energies, n_atoms)`
- Converts to per-atom energies
- Normalizes across molecules of different sizes

### 2. Data Configuration (`mmml/data/loaders.py`)

Extended `DataConfig` class with new parameters:
- `energy_unit`: Input energy unit (default: 'eV')
- `convert_energy_to`: Target unit for conversion (default: None)
- `subtract_atomic_energies`: Enable atomic reference subtraction (default: False)
- `atomic_energy_method`: Method for computing references (default: 'linear_regression')
- `scale_by_atoms`: Enable per-atom scaling (default: False)

Updated `_apply_config_preprocessing()` to apply energy transformations in order:
1. Unit conversion
2. Atomic energy subtraction
3. Per-atom scaling
4. Normalization

All transformations are stored in metadata for reproducibility.

### 3. Training Script (`examples/co2/physnet_train/trainer.py`)

Added command-line arguments:
- `--energy-unit`: Specify input energy unit
- `--convert-energy-to`: Convert to target unit
- `--subtract-atomic-energies`: Enable atomic reference subtraction
- `--atomic-energy-method`: Choose computation method
- `--scale-by-atoms`: Enable per-atom scaling

Added verbose output showing which preprocessing steps are applied.

### 4. Testing and Documentation

#### Test Suite (`examples/co2/physnet_train/test_energy_preprocessing.py`)
Comprehensive tests covering:
- Unit conversion (Hartree ↔ eV)
- Atomic energy computation with varied compositions
- Per-atom scaling
- Full pipeline with real CO2 data

All tests pass ✅

#### Example Scripts
- `train_with_atomic_refs.sh`: Training with atomic energy subtraction
- `train_per_atom.sh`: Training with per-atom scaling

#### Documentation
- `ENERGY_PREPROCESSING.md`: Complete guide with examples and use cases

## Bug Fixes

### NPZ Loading Issue
Fixed `_pickle.UnpicklingError` in `mmml/data/loaders.py`:
- Changed `np.load(file_path, allow_pickle=True)` to `allow_pickle=False`
- The NPZ files contain standard numpy arrays, not pickled objects
- This was causing the loader to incorrectly try to unpickle the files

## Validation Results

### Unit Conversion Test
- Hartree to eV: ✅ Correct (27.211386 eV per Hartree)
- Round-trip conversion: ✅ Preserves values

### Atomic Energy Computation Test
Synthetic data (CO2, CO, O2 molecules):
- Computed C energy: -1000.00 eV (exact match)
- Computed O energy: -2000.00 eV (exact match)
- Binding energies after subtraction: ~0 eV (as expected for ideal gas model)

### Real CO2 Data Results

**Original energies:**
- Mean: -5104.41 eV
- Std: 1.77 eV
- Range: [-5107.89, -5098.45] eV

**With atomic energy subtraction:**
- Mean: 0.00 eV (perfectly centered)
- Std: 1.77 eV (variance preserved)
- Range: [-3.48, 5.96] eV
- Computed atomic energies:
  - C (Z=6): -1020.88 eV
  - O (Z=8): -2041.76 eV

**With per-atom scaling:**
- Mean: -1701.47 eV/atom
- Std: 0.59 eV/atom
- Ratio: 3.00 (consistent with CO2 = 3 atoms)

## Usage Examples

### Basic atomic energy subtraction:
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --subtract-atomic-energies
```

### Unit conversion from Hartree:
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --energy-unit hartree \
  --convert-energy-to eV
```

### Combined preprocessing:
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --subtract-atomic-energies \
  --scale-by-atoms \
  --normalize-energy
```

## Backward Compatibility

✅ All changes are backward compatible:
- Default behavior unchanged (no preprocessing)
- Original training scripts work without modification
- New options are opt-in via command-line flags

## Benefits

1. **Improved Training Stability**: Removing atomic energy offsets reduces the dynamic range of energies, making training more stable

2. **Better Interpretability**: Binding energies are more meaningful than absolute molecular energies

3. **Flexibility**: Support for multiple energy units common in QM calculations

4. **Generalization**: Per-atom scaling helps models generalize across molecule sizes

5. **Reproducibility**: All transformations stored in metadata

## Files Modified

- `mmml/data/preprocessing.py`: Added 4 new functions
- `mmml/data/loaders.py`: Extended DataConfig, updated preprocessing
- `examples/co2/physnet_train/trainer.py`: Added CLI arguments and verbose output

## Files Created

- `examples/co2/physnet_train/test_energy_preprocessing.py`: Test suite
- `examples/co2/physnet_train/train_with_atomic_refs.sh`: Example script
- `examples/co2/physnet_train/train_per_atom.sh`: Example script
- `examples/co2/physnet_train/ENERGY_PREPROCESSING.md`: Documentation

## Testing

Run the test suite:
```bash
cd examples/co2/physnet_train
python test_energy_preprocessing.py
```

Run example training:
```bash
bash train_with_atomic_refs.sh
```

## Future Enhancements

Possible extensions:
1. Support for custom atomic energy dictionaries (e.g., from DFT calculations)
2. Element-specific per-atom normalizations
3. Automatic unit detection from file metadata
4. Integration with other ML potentials (e.g., SchNet, MACE)

