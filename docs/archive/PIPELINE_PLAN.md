# MMML Data Pipeline - Unified Architecture Plan

## Overview
Create a reproducible pipeline from Molpro XML → NPZ → Training → Evaluation for both DCMNet and PhysNetJAX models.

## Current State Analysis

### Existing Components

#### 1. Data Parsers
- ✅ **Molpro XML Parser** (`mmml/parse_molpro/read_molden.py`)
  - Parses: geometry, energies, orbitals, dipoles, gradients, ESP, variables
  - Returns: NumPy arrays in `MolproData` dataclass
  - Status: **Complete and XSD-compliant**

- ✅ **NPZ Readers** 
  - `mmml/physnetjax/physnetjax/data/read_npz.py`
  - `mmml/dcmnet/dcmnet/data.py` (prepare_multiple_datasets)
  - Status: **Working but inconsistent formats**

#### 2. Data Preparation
- `mmml/physnetjax/physnetjax/data/data.py` - Dataset preparation
- `mmml/dcmnet/dcmnet/data.py` - Dataset preparation (similar but different)
- `mmml/physnetjax/physnetjax/data/batches.py` - Batch creation
- Status: **Duplicated logic, needs harmonization**

#### 3. CLI Tools
- `mmml/cli/make_training.py` - Generate training data
- `mmml/cli/run_sim.py` - Run simulations
- `mmml/cli/opt_mmml.py` - Optimization
- `mmml/cli/base.py` - Base CLI functionality
- Status: **Need integration with new parser**

## Proposed Unified Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     MMML Data Pipeline                          │
└─────────────────────────────────────────────────────────────────┘

Step 1: GENERATE
┌──────────────┐
│  Molpro Run  │ → XML files (co2.xml, ...)
└──────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  CLI: mmml molpro-generate                   │
│  - Manages Molpro input generation           │
│  - Batch job submission                      │
│  - Output organization                       │
└──────────────────────────────────────────────┘

Step 2: CONVERT
┌──────────────┐
│  XML Files   │ → read_molden.py → MolproData
└──────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  CLI: mmml xml2npz                           │
│  - Uses parse_molpro/read_molden.py          │
│  - Converts to standardized NPZ format       │
│  - Handles batch conversion                  │
│  - Generates data summary/statistics         │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  Standardized NPZ Format                     │
│  ├─ R: coordinates (n_structures, n_atoms, 3)│
│  ├─ Z: atomic numbers (n_structures, n_atoms)│
│  ├─ E: energies (n_structures,)              │
│  ├─ F: forces (n_structures, n_atoms, 3)     │
│  ├─ D: dipoles (n_structures, 3)             │
│  ├─ esp: ESP values (n_structures, n_grid)   │
│  ├─ esp_grid: grid coords (n_str, n_grid, 3)│
│  ├─ mono: monopoles (n_structures, n_atoms)  │
│  ├─ polar: polarizability (n_str, 3, 3)     │
│  ├─ quadrupole: (n_structures, 3, 3)        │
│  ├─ N: number of atoms (n_structures,)       │
│  └─ metadata: dict with Molpro variables     │
└──────────────────────────────────────────────┘

Step 3: PREPARE
┌──────────────┐
│  NPZ Files   │
└──────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  Unified Data Module                         │
│  mmml/data/__init__.py                       │
│  ├─ load_dataset()                           │
│  ├─ prepare_batches()                        │
│  ├─ train_valid_split()                      │
│  └─ DataConfig (standardized config)         │
└──────────────────────────────────────────────┘
        ↓
┌────────────────┬─────────────────┐
│  DCMNet Data   │  PhysNetJAX Data│
└────────────────┴─────────────────┘

Step 4: TRAIN
┌──────────────────────────────────────────────┐
│  CLI: mmml train                             │
│  - Model selection (dcmnet/physnetjax)       │
│  - Unified training interface                │
│  - Checkpoint management                     │
│  - Logging and monitoring                    │
└──────────────────────────────────────────────┘
        ↓
┌──────────────┐
│  Checkpoints │
└──────────────┘

Step 5: EVALUATE
┌──────────────────────────────────────────────┐
│  CLI: mmml evaluate                          │
│  - Prediction on test sets                   │
│  - Error metrics (MAE, RMSE)                 │
│  - Property-specific analysis                │
│  - Visualization (parity plots, etc.)        │
└──────────────────────────────────────────────┘
        ↓
┌──────────────┐
│  Results     │
│  └─ report/  │
└──────────────┘
```

## Implementation Plan

### Phase 1: Core Data Infrastructure (Week 1)

#### 1.1 Create Unified Data Module
**File**: `mmml/data/__init__.py`

```python
# Unified data loading and preparation
- load_molpro_xml(files) → MolproData
- molpro_to_npz(molpro_data, output_file) → NPZ
- load_npz(file, keys) → dict
- prepare_batches(data, batch_size, model_type) → batches
- DataConfig dataclass for all configuration
```

#### 1.2 Standardize NPZ Format
**File**: `mmml/data/npz_schema.py`

Define canonical NPZ structure:
- Required keys: R, Z, E, N
- Optional keys: F, D, esp, esp_grid, mono, polar, quadrupole
- Metadata: Molpro variables, generation info, units

#### 1.3 Create Molpro → NPZ Converter
**File**: `mmml/data/xml_to_npz.py`

```python
class MolproConverter:
    """Convert Molpro XML to standardized NPZ format"""
    
    def convert_single(xml_file) → dict
    def convert_batch(xml_files, output) → npz
    def generate_statistics(data) → summary
```

### Phase 2: CLI Integration (Week 2)

#### 2.1 XML to NPZ CLI
**File**: `mmml/cli/xml2npz.py`

```bash
mmml xml2npz input_dir/*.xml --output data.npz --validate --summary
```

Features:
- Batch conversion of multiple XML files
- Progress bars (tqdm)
- Validation against schema
- Automatic statistics/summary generation
- Handle failed/incomplete calculations

#### 2.2 Unified Training CLI
**File**: `mmml/cli/train.py`

```bash
mmml train --model dcmnet \
           --data train.npz \
           --valid valid.npz \
           --config config.yaml \
           --output checkpoints/
```

Features:
- Model-agnostic interface (dcmnet, physnetjax)
- Unified configuration format
- Automatic logging and checkpointing
- Resume from checkpoint
- WandB integration

#### 2.3 Evaluation CLI
**File**: `mmml/cli/evaluate.py`

```bash
mmml evaluate --model checkpoints/best.pkl \
              --data test.npz \
              --output results/ \
              --properties E,F,D,esp
```

Features:
- Property-specific metrics
- Visualization generation
- Comparison with reference data
- Export to CSV/JSON

### Phase 3: Data Adapters (Week 2-3)

#### 3.1 DCMNet Adapter
**File**: `mmml/data/adapters/dcmnet.py`

```python
def prepare_dcmnet_batches(data: dict, config: DataConfig) → list:
    """Convert standard format → DCMNet batches"""
    # Handle ESP grid, monopoles, message passing indices
    # Return list of batch dictionaries
```

#### 3.2 PhysNetJAX Adapter
**File**: `mmml/data/adapters/physnetjax.py`

```python
def prepare_physnet_batches(data: dict, config: DataConfig) → list:
    """Convert standard format → PhysNetJAX batches"""
    # Handle forces, periodic boundaries, etc.
    # Return list of batch dictionaries
```

### Phase 4: Documentation & Testing (Week 3)

#### 4.1 Complete Pipeline Example
**File**: `examples/complete_pipeline/README.md`

Full walkthrough:
1. Generate Molpro input
2. Run calculations
3. Convert XML → NPZ
4. Train both models
5. Compare results

#### 4.2 Test Suite
**Files**: `tests/test_pipeline/`

- `test_xml_parsing.py` - Molpro XML parsing
- `test_npz_conversion.py` - XML → NPZ conversion
- `test_data_loading.py` - NPZ → model format
- `test_adapters.py` - Model-specific adapters
- `test_cli.py` - CLI commands

## Data Schema Specification

### Standardized NPZ Keys

```python
REQUIRED_KEYS = {
    'R': 'Coordinates (n_structures, n_atoms, 3) [Angstrom]',
    'Z': 'Atomic numbers (n_structures, n_atoms)',
    'E': 'Energies (n_structures,) [Hartree]',
    'N': 'Number of atoms (n_structures,)',
}

OPTIONAL_KEYS = {
    'F': 'Forces (n_structures, n_atoms, 3) [Hartree/Bohr]',
    'D': 'Dipole moments (n_structures, 3) [Debye]',
    'esp': 'ESP values (n_structures, n_grid)',
    'esp_grid': 'ESP grid coords (n_structures, n_grid, 3)',
    'vdw_surface': 'VDW surface points (n_structures, n_surface, 3)',
    'mono': 'Monopoles (n_structures, n_atoms)',
    'polar': 'Polarizability (n_structures, 3, 3)',
    'quadrupole': 'Quadrupole (n_structures, 3, 3)',
    'n_grid': 'Number of grid points (n_structures,)',
    'com': 'Center of mass (n_structures, 3)',
}

METADATA_KEYS = {
    'molpro_variables': 'Dict of Molpro internal variables',
    'generation_date': 'ISO timestamp',
    'molpro_version': 'Molpro version string',
    'basis_set': 'Basis set used',
    'method': 'Calculation method',
    'units': 'Dict of units for each property',
}
```

## Configuration Schema

### Unified Config YAML

```yaml
# config.yaml
data:
  train_file: data/train.npz
  valid_file: data/valid.npz
  test_file: data/test.npz
  batch_size: 32
  num_workers: 4
  
  # Properties to train on
  targets:
    - energy
    - forces
    - dipole
    - esp
  
  # Data preprocessing
  preprocessing:
    center_coordinates: true
    normalize_energy: false
    esp_mask_vdw: true
    vdw_scale: 1.4

model:
  type: dcmnet  # or physnetjax
  
  # Model-specific parameters
  dcmnet:
    features: 128
    max_degree: 2
    num_iterations: 3
    num_basis_functions: 8
    cutoff: 5.0
    
  physnetjax:
    F: 128
    K: 64
    num_residual_atomic: 2
    cutoff: 10.0

training:
  optimizer: adam
  learning_rate: 0.001
  scheduler: exponential
  max_epochs: 1000
  early_stopping: 50
  
  # Loss weights
  loss_weights:
    energy: 1.0
    forces: 100.0
    dipole: 10.0
    esp: 1.0

logging:
  use_wandb: true
  project: mmml
  log_interval: 10
  checkpoint_interval: 100
```

## Directory Structure

```
mmml/
├── data/                          # NEW: Unified data module
│   ├── __init__.py               # Main data loading API
│   ├── npz_schema.py             # NPZ format specification
│   ├── xml_to_npz.py             # Molpro XML → NPZ converter
│   ├── loaders.py                # Dataset loaders
│   ├── preprocessing.py          # Data preprocessing utilities
│   └── adapters/
│       ├── dcmnet.py             # DCMNet data adapter
│       └── physnetjax.py         # PhysNetJAX data adapter
│
├── parse_molpro/                  # Molpro XML parsing
│   ├── read_molden.py            # ✅ Complete
│   ├── molpro-output.xsd         # ✅ Schema
│   └── example_usage.py          # ✅ Examples
│
├── cli/                           # Command-line interface
│   ├── base.py                   # Base CLI class
│   ├── xml2npz.py                # NEW: XML → NPZ conversion
│   ├── train.py                  # NEW: Unified training
│   ├── evaluate.py               # NEW: Model evaluation
│   ├── make_training.py          # REFACTOR: Use new data module
│   ├── run_sim.py                # REFACTOR: Use new data module
│   └── opt_mmml.py               # REFACTOR: Use new data module
│
├── dcmnet/
│   └── dcmnet/
│       ├── data.py               # REFACTOR: Use mmml.data
│       └── ...
│
├── physnetjax/
│   └── physnetjax/
│       └── data/
│           ├── data.py           # REFACTOR: Use mmml.data
│           └── ...
│
├── tests/
│   ├── test_pipeline/            # NEW: Pipeline tests
│   ├── test_data/                # NEW: Data module tests
│   └── fixtures/                 # Test data
│
├── examples/
│   ├── complete_pipeline/        # NEW: Full pipeline example
│   │   ├── README.md
│   │   ├── 01_generate_data.sh
│   │   ├── 02_convert_xml.sh
│   │   ├── 03_train_models.sh
│   │   └── 04_evaluate.sh
│   └── notebooks/                # Jupyter notebooks
│
├── docs/
│   ├── data_pipeline.md          # Pipeline documentation
│   ├── npz_format.md             # NPZ format specification
│   └── cli_reference.md          # CLI command reference
│
├── config/
│   ├── dcmnet_default.yaml       # Default DCMNet config
│   └── physnetjax_default.yaml   # Default PhysNetJAX config
│
└── PIPELINE_PLAN.md              # This file
```

## Reproducibility Features

### 1. Configuration Management
- YAML-based configuration
- Version control friendly
- Hierarchical configs (base + experiment)
- Automatic config saving with checkpoints

### 2. Data Versioning
- MD5 hashes for datasets
- Metadata tracking in NPZ files
- Data provenance (source XML files)
- Version tags

### 3. Experiment Tracking
- WandB integration
- Automatic logging of:
  - Data statistics
  - Model architecture
  - Training hyperparameters
  - Evaluation metrics
- Checkpoint + config bundling

### 4. Environment Management
- Requirements.txt
- Environment.yml for conda
- Docker container (optional)

## Usage Examples

### Example 1: Complete Pipeline

```bash
# Step 1: Convert Molpro XML to NPZ
mmml xml2npz molpro_outputs/*.xml \
    --output data/qm9_esp.npz \
    --validate \
    --summary data/qm9_summary.json

# Step 2: Train DCMNet
mmml train \
    --model dcmnet \
    --config config/dcmnet_esp.yaml \
    --data data/qm9_esp.npz \
    --output checkpoints/dcmnet_esp/

# Step 3: Train PhysNetJAX (for comparison)
mmml train \
    --model physnetjax \
    --config config/physnet_esp.yaml \
    --data data/qm9_esp.npz \
    --output checkpoints/physnet_esp/

# Step 4: Evaluate both models
mmml evaluate \
    --models checkpoints/dcmnet_esp/best.pkl \
             checkpoints/physnet_esp/best.pkl \
    --data data/test.npz \
    --output results/comparison/ \
    --compare
```

### Example 2: Python API

```python
from mmml.data import load_npz, prepare_batches, DataConfig
from mmml.data.adapters import prepare_dcmnet_batches
from mmml.parse_molpro import read_molpro_xml

# Load and convert
molpro_data = read_molpro_xml('output.xml')
npz_data = molpro_to_npz(molpro_data, 'data.npz')

# Load for training
config = DataConfig(
    batch_size=32,
    targets=['energy', 'forces', 'esp'],
    preprocessing={'center_coordinates': True}
)

train_data = load_npz('train.npz', config)
batches = prepare_dcmnet_batches(train_data, config)

# Train model
model = DCMNet(config.model_params)
train(model, batches, config.training_params)
```

## Success Metrics

### Phase 1 Complete When:
- ✅ Single NPZ format for all models
- ✅ XML → NPZ conversion working
- ✅ Both models can load from standard format
- ✅ Unit tests passing

### Phase 2 Complete When:
- ✅ CLI commands functional
- ✅ Config-driven training works
- ✅ Evaluation pipeline produces reports
- ✅ Integration tests passing

### Phase 3 Complete When:
- ✅ Adapters handle edge cases
- ✅ Performance benchmarks acceptable
- ✅ Memory usage optimized
- ✅ Model comparison tests passing

### Phase 4 Complete When:
- ✅ Complete example runs end-to-end
- ✅ Documentation complete
- ✅ All tests passing (>90% coverage)
- ✅ Ready for production use

## Next Steps

1. **Review this plan** - Get feedback on structure
2. **Create mmml/data/** - Start with core data module
3. **Implement xml2npz** - Build converter
4. **Refactor existing code** - Update DCMNet/PhysNetJAX to use new format
5. **Build CLI** - Create unified command interface
6. **Write tests** - Ensure reproducibility
7. **Document** - Complete examples and docs

## Questions to Resolve

1. **NPZ vs HDF5?** - NPZ is simpler, HDF5 better for very large datasets
2. **Multiple geometries per file?** - For MD trajectories, scans, etc.
3. **Periodic boundaries?** - Do we need to support periodic systems?
4. **Unit conversions?** - Automatic or explicit in config?
5. **ESP grid generation?** - Should converter generate grids or expect them?

---

**Goal**: Enable researchers to go from Molpro calculations to trained models
with a single, well-documented, reproducible pipeline.

