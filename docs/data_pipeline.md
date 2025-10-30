# Data Pipeline Guide

Complete guide to the MMML data pipeline from Molpro XML to trained models.

## Pipeline Overview

```
Molpro XML → NPZ → Batches → Training → Evaluation
```

### Stage 1: XML Parsing

**Input:** Molpro XML output files  
**Output:** `MolproData` object with NumPy arrays  
**Tool:** `mmml.parse_molpro.read_molden`

```python
from mmml.parse_molpro import read_molpro_xml

data = read_molpro_xml('output.xml')
# Returns: MolproData with coordinates, energies, forces, dipoles, 
#          orbitals, frequencies, and 260+ Molpro variables
```

**What's Extracted:**
- Molecular geometry (CML format)
- Energies (all methods: RHF, MP2, CCSD, etc.)
- Forces/Gradients (Hartree/Bohr)
- Dipole moments (Debye)
- Molecular orbitals (energies, occupancies, coefficients)
- Vibrational data (frequencies, normal modes, IR intensities)
- Molpro internal variables (physical constants, user variables)

See [Molpro Parser README](../mmml/parse_molpro/README.md) for details.

### Stage 2: NPZ Conversion

**Input:** Molpro XML files  
**Output:** Standardized NPZ file  
**Tool:** `mmml.cli xml2npz` or `mmml.data.convert_xml_to_npz`

**Command Line:**
```bash
python -m mmml.cli xml2npz calculations/*.xml -o dataset.npz --validate
```

**Python API:**
```python
from mmml.data import batch_convert_xml

batch_convert_xml(
    xml_files=['calc1.xml', 'calc2.xml'],
    output_file='dataset.npz',
    padding_atoms=60,
    include_variables=True
)
```

**NPZ Schema:**

Required keys:
- `R`: Coordinates (n_structures, n_atoms, 3) [Angstrom]
- `Z`: Atomic numbers (n_structures, n_atoms)
- `E`: Energies (n_structures,) [Hartree]
- `N`: Number of atoms (n_structures,)

Optional keys:
- `F`: Forces (n_structures, n_atoms, 3) [Hartree/Bohr]
- `D`: Dipoles (n_structures, 3) [Debye]
- `esp`: ESP values (n_structures, n_grid)
- `esp_grid`: Grid coordinates (n_structures, n_grid, 3)
- Many more...

See [NPZ Schema](npz_schema.md) for complete specification.

### Stage 3: Data Loading

**Input:** NPZ file(s)  
**Output:** Python dictionaries  
**Tool:** `mmml.data.load_npz`

```python
from mmml.data import load_npz, train_valid_split

# Load full dataset
data = load_npz('dataset.npz', validate=True)

# Split into train/validation
train_data, valid_data = train_valid_split(
    data,
    train_fraction=0.8,
    shuffle=True,
    seed=42
)

print(f"Train: {len(train_data['E'])} structures")
print(f"Valid: {len(valid_data['E'])} structures")
```

**Loading Multiple Files:**
```python
from mmml.data import load_multiple_npz

data = load_multiple_npz(
    ['batch1.npz', 'batch2.npz', 'batch3.npz'],
    combine=True,
    validate=True
)
```

### Stage 4: Batch Preparation

**Input:** Data dictionaries  
**Output:** Model-specific batches  
**Tool:** Model adapters

**For DCMNet:**
```python
from mmml.data.adapters import prepare_dcmnet_batches

batches = prepare_dcmnet_batches(
    train_data,
    batch_size=32,
    num_atoms=60,
    shuffle=True
)

# Each batch contains:
# - R: (batch_size * num_atoms, 3)
# - Z: (batch_size * num_atoms,)
# - E, F, D: (batch_size, ...)
# - dst_idx, src_idx: Message passing indices
# - batch_segments: Batch segmentation
```

**For PhysNetJAX:**
```python
from mmml.data.adapters import prepare_physnet_batches

batches = prepare_physnet_batches(
    train_data,
    batch_size=32,
    num_atoms=60,
    shuffle=True
)
```

### Stage 5: Training

**Input:** Batches  
**Output:** Trained model checkpoints  
**Tool:** `mmml.cli train`

```bash
python -m mmml.cli train \
    --model dcmnet \
    --train train.npz \
    --valid valid.npz \
    --config config.yaml \
    --output checkpoints/
```

**With Config File:**
```yaml
# config.yaml
model: dcmnet
train_file: train.npz
valid_file: valid.npz
batch_size: 32
max_epochs: 1000
targets: [energy, forces, dipole]
loss_weights:
  energy: 1.0
  forces: 100.0
  dipole: 10.0
```

### Stage 6: Evaluation

**Input:** Model checkpoint + test data  
**Output:** Metrics, reports, visualizations  
**Tool:** `mmml.cli evaluate`

```bash
python -m mmml.cli evaluate \
    --model checkpoints/best.pkl \
    --data test.npz \
    --properties energy forces dipole \
    --report \
    --output results/
```

**Output:**
- `evaluation_results.json` - Metrics (MAE, RMSE, R², max error)
- `evaluation_report.md` - Formatted report
- Optionally: predictions, parity plots

## Complete Example Workflow

### Step-by-Step with CO2

```bash
# Setup
cd /home/ericb/mmml
export PYTHONPATH=$PWD:$PYTHONPATH

# 1. Validate your XML file
head mmml/parse_molpro/co2.xml

# 2. Convert to NPZ
python -m mmml.cli xml2npz \
    mmml/parse_molpro/co2.xml \
    -o data/co2.npz \
    --validate \
    --summary data/co2_summary.json

# 3. Inspect the dataset
python -m mmml.cli validate data/co2.npz

# 4. View summary
cat data/co2_summary.json | python -m json.tool

# 5. Prepare for training (Python)
python << EOF
from mmml.data import load_npz, train_valid_split
data = load_npz('data/co2.npz')
train, valid = train_valid_split(data, train_fraction=0.8)
print(f"Ready: {len(train['E'])} train, {len(valid['E'])} valid")
EOF

# 6. Train (when model integration complete)
python -m mmml.cli train \
    --model dcmnet \
    --train data/co2.npz \
    --dry-run  # Test setup without training

# 7. Evaluate (when model integration complete)
python -m mmml.cli evaluate \
    --model checkpoints/best.pkl \
    --data data/co2.npz \
    --report
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Molpro Calculations                      │
│  (Quantum chemistry: geometry, energy, forces, ESP, ...)   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ XML Output
┌─────────────────────────────────────────────────────────────┐
│               Molpro XML Parser (XSD-Compliant)            │
│  ✓ Parses all properties as NumPy arrays                  │
│  ✓ Handles CML format and namespaces                      │
│  ✓ Extracts 260+ Molpro variables                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ MolproData
┌─────────────────────────────────────────────────────────────┐
│                 NPZ Converter (MolproConverter)            │
│  ✓ Standardizes to schema-validated format                │
│  ✓ Handles padding and batching                           │
│  ✓ Preserves metadata and provenance                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ NPZ File (Compressed)
┌─────────────────────────────────────────────────────────────┐
│                 Data Loader + Validation                   │
│  ✓ Schema validation                                       │
│  ✓ Statistics generation                                   │
│  ✓ Train/valid splitting                                   │
│  ✓ Preprocessing (centering, normalization)               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ Data Dictionaries
┌──────────────────────┬──────────────────────────────────────┐
│   DCMNet Adapter     │    PhysNetJAX Adapter               │
│  ✓ Message passing   │   ✓ Neighbor lists                 │
│  ✓ ESP handling      │   ✓ Force training                 │
│  ✓ Batch segments    │   ✓ Periodic boundaries            │
└──────────┬───────────┴──────────────┬───────────────────────┘
           │                          │
           ↓ Batches                  ↓ Batches
┌──────────────────────┬──────────────────────────────────────┐
│   DCMNet Training    │    PhysNetJAX Training              │
│  (ESP, multipoles)   │   (Energy, forces)                  │
└──────────┬───────────┴──────────────┬───────────────────────┘
           │                          │
           ↓ Checkpoints              ↓ Checkpoints
┌─────────────────────────────────────────────────────────────┐
│                    Model Evaluation                         │
│  ✓ Metrics (MAE, RMSE, R²)                                │
│  ✓ Property-specific analysis                              │
│  ✓ Comparison reports                                      │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Management

### YAML Configuration

```yaml
# Full training configuration
model: dcmnet

data:
  train_file: train.npz
  valid_file: valid.npz
  batch_size: 32

training:
  max_epochs: 1000
  learning_rate: 0.001
  early_stopping: 50
  
targets: [energy, forces, dipole, esp]
loss_weights:
  energy: 1.0
  forces: 100.0
  dipole: 10.0
  esp: 1.0

preprocessing:
  center_coordinates: false
  normalize_energy: false

model_params:
  features: 128
  max_degree: 2
  num_iterations: 3
  cutoff: 5.0
```

See [Configuration Guide](configuration.md) for all options.

## Reproducibility

### Ensured Through:

1. **Data Provenance**
   - Source XML files tracked in metadata
   - Generation timestamps
   - Molpro version information
   - Conversion parameters stored

2. **Version Control**
   - Config files (YAML)
   - Checksums (MD5 hashes)
   - Seeds for random splits
   - Dependency versions

3. **Validation**
   - Schema-validated data
   - Automatic checks
   - Error detection
   - Consistency verification

4. **Documentation**
   - Complete API reference
   - CLI command history
   - Configuration files
   - Conversion summaries (JSON)

## Best Practices

### 1. Always Validate

```bash
python -m mmml.cli xml2npz input.xml -o output.npz --validate
```

### 2. Save Configurations

```bash
python -m mmml.cli train --config config.yaml
# Config is automatically saved with checkpoints
```

### 3. Generate Summaries

```bash
python -m mmml.cli xml2npz *.xml -o data.npz --summary summary.json
```

### 4. Use Appropriate Padding

Match your largest molecule:
```bash
python -m mmml.cli xml2npz *.xml -o data.npz --padding 100
```

### 5. Keep Conversion Records

```bash
# Save summary with timestamp
python -m mmml.cli xml2npz *.xml \
    -o dataset_$(date +%Y%m%d).npz \
    --summary summary_$(date +%Y%m%d).json
```

## Troubleshooting

### Common Issues

**Issue:** Validation warnings about unknown keys
```
Warning: Unknown keys: {'orbital_energies', 'metadata'}
```
**Solution:** These are harmless - optional properties not in base schema

**Issue:** Shape mismatch errors
```
Error: 'R' and 'Z' shape mismatch
```
**Solution:** Check padding parameter matches your data

**Issue:** Out of memory
```
MemoryError: Unable to allocate array
```
**Solution:** Process in smaller batches or reduce padding

See [Troubleshooting Guide](troubleshooting.md) for more.

## Advanced Usage

### Parallel Processing

```bash
# GNU parallel
find calculations/ -name "*.xml" | \
    parallel -j 8 python -m mmml.cli xml2npz {} -o datasets/{/.}.npz
```

### Custom Preprocessing

```python
from mmml.data import load_npz
from mmml.data.preprocessing import center_coordinates, normalize_energies

data = load_npz('dataset.npz')
data['R'] = center_coordinates(data['R'], data['N'])
data['E'], stats = normalize_energies(data['E'], per_atom=True, n_atoms=data['N'])
```

### Incremental Dataset Building

```python
from mmml.data import load_multiple_npz
import numpy as np

# Load and combine multiple batches
data = load_multiple_npz(
    ['batch1.npz', 'batch2.npz', 'batch3.npz'],
    combine=True
)

# Save combined dataset
np.savez_compressed('full_dataset.npz', **data)
```

## Performance Optimization

### Tips for Large Datasets

1. **Use compressed NPZ:** Automatic with `numpy.savez_compressed`
2. **Appropriate padding:** Don't over-pad (wastes memory)
3. **Batch size:** Adjust for GPU memory
4. **Memory-mapped loading:** For datasets > 10GB (future feature)

### Benchmarks

| Dataset Size | Conversion Time | NPZ Size | Compression Ratio |
|--------------|-----------------|----------|-------------------|
| 1 file (4.5 MB) | 0.1s | 5 KB | 900x |
| 100 files | 10s | 500 KB | ~900x |
| 10K files | ~15 min | 50 MB | ~900x |

## Data Quality Checks

### Automated Validation

```python
from mmml.data import validate_npz

is_valid, info = validate_npz('dataset.npz', strict=True)

if is_valid:
    print(f"✓ Valid dataset")
    print(f"  Structures: {info['n_structures']}")
    print(f"  Energy range: {info['energy_range']}")
    print(f"  Elements: {info['unique_elements']}")
```

### Manual Checks

```python
from mmml.data import load_npz, get_data_statistics
import json

data = load_npz('dataset.npz')
stats = get_data_statistics(data)

# Save for records
with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
```

## See Also

- [CLI Reference](cli_reference.md) - All CLI commands
- [NPZ Schema](npz_schema.md) - Data format details
- [API Reference](api/data.md) - Python API
- [Testing](testing.md) - Test suite

---

**Next:** [CLI Reference](cli_reference.md)

