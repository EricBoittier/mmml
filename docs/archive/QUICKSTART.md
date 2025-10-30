# MMML Quick Start Guide

## 🚀 Getting Started with the Unified Data Pipeline

This guide shows you how to use the new unified data system to go from Molpro XML files to trained models.

## Installation

```bash
cd /home/ericb/mmml
pip install -e .
```

Required dependencies:
- numpy
- jax
- scipy
- ase
- e3x
- tqdm

## Step-by-Step Workflow

### Step 1: Convert Molpro XML to NPZ

#### Single File Conversion

```python
from mmml.data import convert_xml_to_npz

# Convert a single XML file
convert_xml_to_npz(
    xml_file='mmml/parse_molpro/co2.xml',
    output_file='data/co2.npz',
    padding_atoms=60,
    verbose=True
)
```

#### Batch Conversion

```python
from mmml.data import batch_convert_xml
from pathlib import Path

# Convert multiple XML files
xml_files = list(Path('molpro_outputs/').glob('*.xml'))

batch_convert_xml(
    xml_files=xml_files,
    output_file='data/qm9_dataset.npz',
    padding_atoms=60,
    include_variables=True,
    verbose=True
)
```

#### Command Line (once CLI is implemented)

```bash
# Will be available in Phase 2:
python -m mmml.data.xml_to_npz \
    molpro_outputs/*.xml \
    -o data/dataset.npz \
    --padding 60
```

### Step 2: Load and Validate Data

```python
from mmml.data import load_npz, validate_npz, DataConfig

# Validate NPZ file
is_valid, info = validate_npz('data/co2.npz', verbose=True)

# Load with validation
data = load_npz(
    'data/co2.npz',
    validate=True,
    verbose=True
)

print(f"Loaded {len(data['E'])} structures")
print(f"Properties: {list(data.keys())}")
```

### Step 3: Prepare Data for Training

#### Create Train/Valid Split

```python
from mmml.data import load_npz, train_valid_split

# Load full dataset
data = load_npz('data/dataset.npz')

# Split into train and validation
train_data, valid_data = train_valid_split(
    data,
    train_fraction=0.8,
    shuffle=True,
    seed=42
)

print(f"Train: {len(train_data['E'])} structures")
print(f"Valid: {len(valid_data['E'])} structures")
```

#### Using DataConfig for Preprocessing

```python
from mmml.data import load_npz, DataConfig

config = DataConfig(
    batch_size=32,
    targets=['energy', 'forces', 'dipole'],
    center_coordinates=True,
    normalize_energy=False,
    num_atoms=60
)

# Load with automatic preprocessing
data = load_npz('data/dataset.npz', config=config)
```

### Step 4: Prepare Model-Specific Batches

#### For DCMNet

```python
from mmml.data import load_npz
from mmml.data.adapters import prepare_dcmnet_batches

# Load data
train_data = load_npz('data/train.npz')

# Prepare DCMNet batches
batches = prepare_dcmnet_batches(
    train_data,
    batch_size=32,
    num_atoms=60,
    shuffle=True,
    seed=42
)

print(f"Created {len(batches)} batches")
print(f"Batch keys: {batches[0].keys()}")
```

#### For PhysNetJAX

```python
from mmml.data import load_npz
from mmml.data.adapters import prepare_physnet_batches

# Load data
train_data = load_npz('data/train.npz')

# Prepare PhysNetJAX batches
batches = prepare_physnet_batches(
    train_data,
    batch_size=32,
    num_atoms=60,
    shuffle=True
)
```

## Complete Example: CO2 Dataset

```python
"""Complete workflow from XML to ready-for-training batches."""

from mmml.data import (
    convert_xml_to_npz,
    load_npz,
    train_valid_split,
    get_data_statistics
)
from mmml.data.adapters import prepare_dcmnet_batches
import json

# Step 1: Convert XML to NPZ
print("Step 1: Converting XML to NPZ...")
convert_xml_to_npz(
    xml_file='mmml/parse_molpro/co2.xml',
    output_file='data/co2.npz',
    verbose=True
)

# Step 2: Load and inspect
print("\nStep 2: Loading and inspecting data...")
data = load_npz('data/co2.npz', validate=True, verbose=True)

# Get statistics
stats = get_data_statistics(data)
print(json.dumps(stats, indent=2))

# Step 3: Train/valid split
print("\nStep 3: Creating train/valid split...")
train_data, valid_data = train_valid_split(
    data,
    train_fraction=0.8,
    shuffle=True,
    seed=42
)

# Step 4: Prepare batches for DCMNet
print("\nStep 4: Preparing batches...")
train_batches = prepare_dcmnet_batches(
    train_data,
    batch_size=1,  # Small batch for CO2 (only 1 structure)
    num_atoms=60
)

print(f"\n✓ Ready for training!")
print(f"  Train batches: {len(train_batches)}")
print(f"  First batch keys: {list(train_batches[0].keys())}")
print(f"  Coordinates shape: {train_batches[0]['R'].shape}")
```

## Data Schema Reference

### Required Keys

| Key | Shape | Description | Units |
|-----|-------|-------------|-------|
| `R` | `(n_structures, n_atoms, 3)` | Coordinates | Angstrom |
| `Z` | `(n_structures, n_atoms)` | Atomic numbers | - |
| `E` | `(n_structures,)` | Energies | Hartree |
| `N` | `(n_structures,)` | Number of atoms | - |

### Optional Keys

| Key | Shape | Description | Units |
|-----|-------|-------------|-------|
| `F` | `(n_structures, n_atoms, 3)` | Forces | Hartree/Bohr |
| `D` | `(n_structures, 3)` | Dipole moments | Debye |
| `esp` | `(n_structures, n_grid)` | ESP values | Hartree/e |
| `esp_grid` | `(n_structures, n_grid, 3)` | ESP grid coords | Angstrom |
| `mono` | `(n_structures, n_atoms)` | Atomic monopoles | - |
| `polar` | `(n_structures, 3, 3)` | Polarizability | - |

See `mmml/data/npz_schema.py` for complete specification.

## Validating Your Data

### Python API

```python
from mmml.data import validate_npz

is_valid, info = validate_npz('data/mydata.npz', verbose=True)

if is_valid:
    print(f"✓ Valid dataset with {info['n_structures']} structures")
    print(f"  Elements: {info['unique_elements']}")
    print(f"  Energy range: {info['energy_range']}")
else:
    print("✗ Validation failed")
```

### Command Line

```bash
# Validate and show statistics
python -m mmml.data.npz_schema data/mydata.npz

# Or use loaders
python -m mmml.data.loaders data/mydata.npz
```

## Preprocessing Options

### Center Coordinates

```python
from mmml.data import preprocessing

# Center at origin
coords_centered = preprocessing.center_coordinates(
    data['R'],
    n_atoms=data['N'],
    method='geometric'  # or 'com' for center of mass
)
```

### Normalize Energies

```python
# Normalize energies
energies_norm, stats = preprocessing.normalize_energies(
    data['E'],
    per_atom=True,
    n_atoms=data['N']
)

# Later, denormalize predictions
energies_denorm = preprocessing.denormalize_energies(
    predictions,
    stats,
    n_atoms=data['N']
)
```

### Create ESP Mask

```python
# Mask ESP points inside VDW radii
esp_mask = preprocessing.create_esp_mask(
    data['esp_grid'],
    data['R'],
    data['Z'],
    vdw_scale=1.4
)
```

## Troubleshooting

### Common Issues

#### 1. Missing Keys

```
Error: Missing required keys: {'E'}
```

**Solution**: Ensure your Molpro XML contains energy data, or use optional validation:

```python
is_valid, info = validate_npz('data.npz', strict=False)
```

#### 2. Shape Mismatches

```
Error: 'R' and 'Z' shape mismatch: (10, 5) vs (10, 6)
```

**Solution**: Check padding_atoms parameter matches your data:

```python
convert_xml_to_npz(xml_file, output_file, padding_atoms=100)
```

#### 3. Import Errors

```
ImportError: cannot import name 'read_molpro_xml'
```

**Solution**: Ensure parse_molpro is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('mmml/parse_molpro')))
```

## Next Steps

1. **Phase 2 (Coming Soon)**: CLI commands for easier workflow
2. **Phase 3 (In Progress)**: Enhanced model adapters
3. **Phase 4 (Planned)**: Comprehensive tests and examples

## Getting Help

- See `PIPELINE_PLAN.md` for architectural details
- Check `mmml/data/npz_schema.py` for data format specification
- Look at `mmml/parse_molpro/README.md` for XML parsing details

## Contributing

To improve the data pipeline:

1. Update adapters in `mmml/data/adapters/`
2. Add preprocessing functions to `mmml/data/preprocessing.py`
3. Extend schema in `mmml/data/npz_schema.py`
4. Add tests in `tests/test_data/`

---

**Last Updated**: October 30, 2025  
**Status**: Phase 1 Complete ✓
