# Energy Preprocessing Options

This document describes the energy preprocessing options available in the PhysNetJax training pipeline.

## Overview

The training pipeline now supports comprehensive energy preprocessing to handle different energy scales, units, and reference states. These options can be combined to prepare your data optimally for training.

## Available Options

### 1. Energy Unit Conversion

Convert energy units between different scales.

**Command-line flags:**
- `--energy-unit`: Specify the input energy unit (default: `eV`)
- `--convert-energy-to`: Convert to a target unit

**Supported units:**
- `eV` (electron volts)
- `hartree` (atomic units)
- `kcal/mol` (kilocalories per mole)
- `kJ/mol` (kilojoules per mole)

**Example:**
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --energy-unit hartree \
  --convert-energy-to eV
```

### 2. Atomic Energy Reference Subtraction

Remove atomic energy contributions from molecular energies. This is useful when you want the model to learn binding/interaction energies rather than total energies.

**Command-line flags:**
- `--subtract-atomic-energies`: Enable atomic energy subtraction
- `--atomic-energy-method`: Method for computing references (default: `linear_regression`)

**Methods:**
- `linear_regression`: Fit atomic energies using least-squares regression on the training set
- `mean`: Use mean per-atom energies

**Example:**
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression
```

**How it works:**

For each molecule: `E_binding = E_molecular - Σ(E_atomic_i)`

The atomic energies are computed automatically from your training data using linear regression:
```
E_molecular = n_C * E_C + n_O * E_O + n_H * E_H + ...
```

This transforms the learning problem from predicting absolute energies (e.g., -5104 eV for CO2) to predicting binding/interaction energies (e.g., -10 eV).

**Benefits:**
- Removes large absolute energy offsets
- Focuses model on learning chemical interactions
- Can improve training stability and convergence
- Makes energies more interpretable (binding energies)

### 3. Per-Atom Energy Scaling

Scale energies by the number of atoms in each molecule. This normalizes energies to be comparable across molecules of different sizes.

**Command-line flag:**
- `--scale-by-atoms`: Enable per-atom scaling

**Example:**
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --scale-by-atoms
```

**How it works:**

For each molecule: `E_per_atom = E_molecular / N_atoms`

**Benefits:**
- Normalizes energies across different molecular sizes
- Useful for datasets with varying numbers of atoms
- Can improve model generalization

### 4. Energy Normalization

Normalize energies to mean=0 and std=1. This standardizes the energy scale for training.

**Command-line flag:**
- `--normalize-energy`: Enable normalization

**Example:**
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --normalize-energy
```

## Combining Options

All preprocessing options can be combined. They are applied in the following order:

1. **Unit conversion** (if requested)
2. **Atomic energy subtraction** (if requested)
3. **Per-atom scaling** (if requested)
4. **Normalization** (if requested)

**Example combining all options:**
```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --energy-unit hartree \
  --convert-energy-to eV \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression \
  --scale-by-atoms \
  --normalize-energy
```

## Use Cases

### Case 1: Learning Binding Energies

For learning molecular binding/interaction energies while ignoring atomic contributions:

```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --subtract-atomic-energies \
  --normalize-energy
```

This will:
1. Compute atomic energy references from training data
2. Subtract them from all molecular energies
3. Normalize the resulting binding energies

### Case 2: Mixed-Size Molecules

For datasets with molecules of varying sizes:

```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --scale-by-atoms \
  --normalize-energy
```

This ensures all molecules contribute equally to training regardless of size.

### Case 3: Unit Conversion from QM Data

When your QM calculations are in Hartree but you want to train in eV:

```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --energy-unit hartree \
  --convert-energy-to eV
```

## Metadata

When preprocessing is applied, the transformations are stored in the data metadata:

```python
metadata = {
    'energy_converted': {'from': 'hartree', 'to': 'eV'},
    'atomic_energies': {6: -1020.88, 8: -2041.76},  # C, O
    'atomic_energy_method': 'linear_regression',
    'energy_scaled_by_atoms': True,
    'energy_normalization': {'mean': 0.0, 'std': 1.77}
}
```

This allows you to reverse transformations for predictions and analysis.

## Example Results

### CO2 Dataset

Original energies:
- Mean: -5104.41 eV
- Std: 1.77 eV
- Range: [-5107.89, -5098.45] eV

After atomic energy subtraction:
- Mean: 0.00 eV
- Std: 1.77 eV  (unchanged, as expected)
- Range: [-3.48, 5.96] eV
- Atomic energies: C = -1020.88 eV, O = -2041.76 eV

After per-atom scaling:
- Mean: -1701.47 eV/atom
- Std: 0.59 eV/atom
- Ratio: 3.00 (consistent with CO2 having 3 atoms)

## Testing

Run the test suite to verify preprocessing functionality:

```bash
python test_energy_preprocessing.py
```

This runs comprehensive tests including:
1. Unit conversion (Hartree ↔ eV)
2. Atomic energy computation and subtraction
3. Per-atom scaling
4. Full pipeline with real CO2 data

## References

- [PhysNet Paper](https://doi.org/10.1021/acs.jctc.8b00908) - Discussion of atomic energy references
- [SchNet Paper](https://arxiv.org/abs/1706.08566) - Atomic energy decomposition in neural networks

