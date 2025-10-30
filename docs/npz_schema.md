# NPZ Data Format Specification

Complete specification of the standardized NPZ format used across all MMML models.

## Overview

The MMML NPZ format is a schema-validated, compressed NumPy archive that stores molecular data in a standardized way compatible with DCMNet, PhysNetJAX, and future models.

## Design Principles

1. **Schema-Driven** - Validated against defined specification
2. **Self-Documenting** - Includes metadata and units
3. **Extensible** - Optional keys for additional properties
4. **Efficient** - Compressed storage, ~900x compression typical
5. **Portable** - Standard NumPy format, cross-platform

## Required Keys

These keys **must** be present in every NPZ file:

| Key | Shape | Type | Units | Description |
|-----|-------|------|-------|-------------|
| `R` | `(n_structures, n_atoms, 3)` | float64 | Angstrom | Cartesian coordinates |
| `Z` | `(n_structures, n_atoms)` | int32 | - | Atomic numbers |
| `E` | `(n_structures,)` or `(n_structures, 1)` | float64 | Hartree | Total energies |
| `N` | `(n_structures,)` | int32 | - | Number of atoms per structure |

### Required Key Details

#### R - Coordinates
- **Format:** 3D array with last dimension [x, y, z]
- **Units:** Angstrom (as output by Molpro)
- **Padding:** Padded to fixed `n_atoms` (typically 60)
- **Valid atoms:** First `N[i]` atoms are valid, rest are padding (zeros)

```python
# Example
R.shape  # (1000, 60, 3)
N[0]  # 15 (actual atoms)
R[0, :15, :]  # Valid coordinates
R[0, 15:, :]  # Padding (all zeros)
```

#### Z - Atomic Numbers
- **Format:** 2D array of integers
- **Range:** 1-118 (periodic table)
- **Padding:** Padded with zeros
- **Common:** 1=H, 6=C, 7=N, 8=O

```python
# Example
Z[0]  # array([6, 8, 8, 1, 1, 0, 0, ...])
# Carbon, Oxygen, Oxygen, Hydrogen, Hydrogen, padding...
```

#### E - Energies
- **Format:** 1D or 2D array
- **Units:** Hartree (atomic units)
- **Type:** Total electronic energy (not per-atom)
- **Method:** Best available (RHF, MP2, CCSD in order of preference)

```python
# Example
E.shape  # (1000,) or (1000, 1)
E[0]  # -187.571314 (Hartree)
```

#### N - Number of Atoms
- **Format:** 1D array of integers
- **Purpose:** Track actual vs. padded atoms
- **Range:** 1 to n_atoms (padding size)

```python
# Example
N[0]  # 15 (this structure has 15 atoms)
```

## Optional Keys

These keys **may** be present:

### Molecular Properties

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `F` | `(n_structures, n_atoms, 3)` | Hartree/Bohr | Forces (negative gradient) |
| `D` | `(n_structures, 3)` | Debye | Dipole moment vector |
| `Dxyz` | `(n_structures, 3)` | Debye | Dipole components (alias for D) |
| `mono` | `(n_structures, n_atoms)` | e | Atomic monopoles/charges |
| `polar` | `(n_structures, 3, 3)` | a.u. | Polarizability tensor |
| `quadrupole` | `(n_structures, 3, 3)` | ea₀² | Quadrupole tensor |

### Electrostatic Potential

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `esp` | `(n_structures, n_grid)` | Hartree/e | ESP values at grid points |
| `esp_grid` | `(n_structures, n_grid, 3)` | Angstrom | ESP grid point coordinates |
| `espMask` | `(n_structures, n_grid)` | bool | VDW exclusion mask |
| `vdw_surface` | `(n_structures, n_surface, 3)` | Angstrom | VDW surface points |
| `n_grid` | `(n_structures,)` | int | Number of grid points |

### Quantum Chemistry Data

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `orbital_energies` | `(n_structures, n_orbitals)` | Hartree | MO energies |
| `orbital_occupancies` | `(n_structures, n_orbitals)` | - | MO occupation numbers |
| `frequencies` | `(n_structures, n_modes)` | cm⁻¹ | Vibrational frequencies |
| `ir_intensities` | `(n_structures, n_modes)` | km/mol | IR intensities |

### Utilities

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `id` | `(n_structures,)` | str/int | Structure identifiers |
| `com` | `(n_structures, 3)` | Angstrom | Center of mass |
| `metadata` | `(1,)` | object | Metadata dictionary (pickled) |

## Metadata

Stored in `metadata` key as pickled dictionary:

```python
metadata = {
    'generation_date': '2025-10-30T12:00:00',
    'source_files': ['calc1.xml', 'calc2.xml'],
    'molpro_variables': {...},  # 260+ variables
    'units': {
        'R': 'Angstrom',
        'E': 'Hartree',
        'F': 'Hartree/Bohr',
        'D': 'Debye'
    },
    'energies_by_method': {
        'RHF': -187.571,
        'MP2': -188.123
    },
    'molpro_version': '2025.2',
    'conversion_info': {...}
}
```

## Validation

### Schema Validation

```python
from mmml.data import validate_npz

is_valid, info = validate_npz('dataset.npz', strict=False)
```

**Checks:**
- ✓ Required keys present
- ✓ Array shapes compatible
- ✓ Data types correct
- ✓ N values ≤ array dimensions
- ✓ R and Z shapes match
- ✓ F and R shapes match (if F present)
- ⚠ Unknown keys (warning only)

### Manual Validation

```python
import numpy as np

data = np.load('dataset.npz', allow_pickle=True)

# Check required keys
assert 'R' in data
assert 'Z' in data
assert 'E' in data
assert 'N' in data

# Check shapes
n_structures, n_atoms, _ = data['R'].shape
assert data['Z'].shape == (n_structures, n_atoms)
assert len(data['E']) == n_structures
assert len(data['N']) == n_structures

# Check values
assert np.all(data['N'] <= n_atoms)
assert np.all(data['Z'] >= 0)
assert np.all(data['Z'] <= 118)  # Max atomic number
```

## Unit Conventions

### Lengths
- **Angstrom** (Å): All coordinates (R, esp_grid, com)
- **Bohr** (a₀): Force denominators only

### Energies
- **Hartree** (Eₕ): All energies and ESP values
- **kcal/mol**: Not used in NPZ (convert in preprocessing)

### Properties
- **Debye** (D): Dipole moments
- **cm⁻¹**: Vibrational frequencies
- **e**: Charges (monopoles)

### Conversions

```python
# Common conversions
HARTREE_TO_KCALMOL = 627.509
BOHR_TO_ANGSTROM = 0.529177
DEBYE_TO_AU = 0.393430

# Example: Energy in kcal/mol
E_kcalmol = data['E'] * HARTREE_TO_KCALMOL
```

## File Size Guidelines

| Dataset | Structures | Atoms | Compressed Size | Uncompressed |
|---------|------------|-------|-----------------|--------------|
| Small | 100 | 20 | ~500 KB | ~5 MB |
| Medium | 1,000 | 60 | ~5 MB | ~50 MB |
| Large | 10,000 | 60 | ~50 MB | ~500 MB |
| XL | 100,000 | 100 | ~1 GB | ~10 GB |

**Compression:** ~10x typical, up to 900x for sparse data

## Creating NPZ Files

### From Molpro XML

```bash
python -m mmml.cli xml2npz calculations/*.xml -o dataset.npz
```

### From Python

```python
import numpy as np

data = {
    'R': coordinates,  # (n, n_atoms, 3)
    'Z': atomic_numbers,  # (n, n_atoms)
    'E': energies,  # (n,)
    'N': n_atoms_per_structure,  # (n,)
}

np.savez_compressed('dataset.npz', **data)
```

### From Other Formats

```python
# From ASE (future feature)
from mmml.data.converters import ase_to_npz
ase_to_npz(atoms_list, 'dataset.npz')

# From XYZ (future feature)
from mmml.data.converters import xyz_to_npz
xyz_to_npz('structures.xyz', 'dataset.npz')
```

## Reading NPZ Files

### Command Line

```bash
python -m mmml.cli validate dataset.npz
```

### Python - Full Data

```python
from mmml.data import load_npz

data = load_npz('dataset.npz', validate=True)
print(f"Keys: {data.keys()}")
print(f"Structures: {len(data['E'])}")
```

### Python - Specific Keys

```python
data = load_npz('dataset.npz', keys=['R', 'Z', 'E'])
```

### NumPy Directly

```python
import numpy as np

data = np.load('dataset.npz', allow_pickle=True)
coordinates = data['R']
energies = data['E']
metadata = data['metadata'][0]  # Unpickle metadata dict
```

## Best Practices

### 1. Always Compress

```python
# Good: Compressed
np.savez_compressed('data.npz', **data)

# Bad: Uncompressed (10x larger)
np.savez('data.npz', **data)
```

### 2. Include Metadata

```python
data['metadata'] = np.array([{
    'source': 'molpro_calculations',
    'date': '2025-10-30',
    'units': {'R': 'Angstrom', 'E': 'Hartree'}
}], dtype=object)
```

### 3. Validate Before Use

```python
from mmml.data import validate_npz

is_valid, _ = validate_npz('dataset.npz')
if not is_valid:
    raise ValueError("Invalid dataset!")
```

### 4. Use Appropriate Padding

```python
# Check your max molecule size
max_atoms = max(n_atoms_per_molecule)

# Pad with ~20% buffer
padding = int(max_atoms * 1.2)
```

### 5. Document Changes

```python
# Add to metadata
data['metadata'][0]['modifications'] = [
    'Centered coordinates',
    'Normalized energies',
    'Filtered failed calculations'
]
```

## Schema Evolution

### Version 0.1.0 (Current)

Initial schema with required and optional keys as documented above.

### Future Versions

Planned additions:
- Periodic boundary conditions
- Multiple conformers per structure
- Time series (MD trajectories)
- Additional quantum properties
- Sparse data support

**Backward Compatibility:** Guaranteed for required keys.

## Extending the Schema

### Adding Custom Keys

```python
# Custom keys are allowed (with warning)
data['custom_property'] = my_custom_data
np.savez_compressed('dataset.npz', **data)

# Document in metadata
data['metadata'][0]['custom_keys'] = {
    'custom_property': 'Description of my custom property'
}
```

### Proposing Schema Changes

1. Open GitHub issue with proposal
2. Provide use case and example data
3. Ensure backward compatibility
4. Update schema documentation
5. Add validation tests

## See Also

- [Data Pipeline](data_pipeline.md) - Complete pipeline
- [API Reference](api/data.md) - Loading and conversion API
- [Examples](../examples/) - Example datasets

---

**Schema Version:** 0.1.0  
**Last Updated:** October 30, 2025

