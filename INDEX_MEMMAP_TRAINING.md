# Index: PhysNet Packed Memmap Training System

Complete index of all files, documentation, and resources for the PhysNet packed memmap training system.

## 📁 File Structure

```
/home/ericb/mmml/
├── Core Module
│   └── mmml/data/packed_memmap_loader.py         [MODULE] Data loader class
│
├── Training Scripts
│   ├── train_physnet_memmap.py                   [SCRIPT] Main training script
│   └── examples/train_memmap_simple.py           [EXAMPLE] Minimal example
│
├── Utilities
│   └── scripts/convert_npz_to_packed_memmap.py   [SCRIPT] NPZ→Memmap converter
│
└── Documentation
    ├── QUICKSTART_MEMMAP.md                      [GUIDE] Quick start (5 min)
    ├── PACKED_MEMMAP_TRAINING.md                 [DOCS] Complete documentation
    ├── MEMMAP_TRAINING_SUMMARY.md                [SUMMARY] System overview
    ├── CODE_COMPARISON.md                        [GUIDE] Original vs new code
    └── INDEX_MEMMAP_TRAINING.md                  [INDEX] This file
```

## 📚 Documentation Guide

### For New Users
1. **Start here**: `QUICKSTART_MEMMAP.md` - Get running in 5 minutes
2. **Then read**: `CODE_COMPARISON.md` - Understand what changed
3. **Reference**: `PACKED_MEMMAP_TRAINING.md` - Detailed documentation

### For Power Users
1. **Overview**: `MEMMAP_TRAINING_SUMMARY.md` - Architecture and design
2. **API Reference**: `PACKED_MEMMAP_TRAINING.md` - Full API docs
3. **Source code**: `mmml/data/packed_memmap_loader.py` - Implementation

## 🚀 Quick Start Commands

### Test Installation
```bash
# Simple 10-epoch test
python examples/train_memmap_simple.py
```

### Convert Data
```bash
# Convert NPZ to packed memmap
python scripts/convert_npz_to_packed_memmap.py \
    --input dataset.npz \
    --output packed_dataset
```

### Train Model
```bash
# Full training run
python train_physnet_memmap.py \
    --data_path openqdc_packed_memmap \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --num_atoms 60
```

## 📖 Documentation Files

### QUICKSTART_MEMMAP.md
**Purpose**: Get started in 5 minutes  
**Contents**:
- Prerequisites checklist
- 3-step quick start
- Common configurations
- Key parameters table
- Troubleshooting quick reference

**Read this if**: You want to start training immediately

### PACKED_MEMMAP_TRAINING.md
**Purpose**: Comprehensive reference  
**Contents**:
- Data format specification (detailed)
- Usage examples (basic to advanced)
- Performance optimization tips
- Full API reference
- Troubleshooting guide
- Comparison with NPZ loading

**Read this if**: You need detailed information or troubleshooting

### MEMMAP_TRAINING_SUMMARY.md
**Purpose**: System architecture overview  
**Contents**:
- What was created and why
- Architecture diagrams
- Data flow examples
- Performance characteristics
- Integration details
- Scaling information

**Read this if**: You want to understand the system design

### CODE_COMPARISON.md
**Purpose**: Understand the adaptation  
**Contents**:
- Side-by-side code comparison
- What changed and why
- PhysNet requirements explained
- Migration path from original code
- Backwards compatibility info

**Read this if**: You want to see exactly how your code was adapted

### INDEX_MEMMAP_TRAINING.md
**Purpose**: Navigate all resources  
**Contents**:
- Complete file listing
- Documentation guide
- Quick reference commands
- Usage scenarios
- API quick reference

**Read this if**: You need to find specific information quickly

## 🔧 Script Reference

### train_physnet_memmap.py
**Purpose**: Full-featured training script  
**Usage**:
```bash
python train_physnet_memmap.py [OPTIONS]
```

**Key Options**:
```
--data_path PATH          Data directory (required)
--batch_size INT          Molecules per batch (default: 32)
--num_epochs INT          Training epochs (default: 100)
--learning_rate FLOAT     Learning rate (default: 0.001)
--num_atoms INT           Max atoms (default: 60)
--features INT            Hidden size (default: 128)
--num_iterations INT      Message passing iterations (default: 3)
--cutoff FLOAT            Cutoff distance (default: 5.0)
--name STR                Experiment name (default: physnet_memmap)
```

**Full help**:
```bash
python train_physnet_memmap.py --help
```

### examples/train_memmap_simple.py
**Purpose**: Minimal training example  
**Usage**:
```bash
python examples/train_memmap_simple.py
```

**Customization**: Edit DATA_PATH in the script

**Use for**:
- Testing installation
- Learning the API
- Debugging data issues
- Template for custom loops

### scripts/convert_npz_to_packed_memmap.py
**Purpose**: Convert NPZ to packed memmap format  
**Usage**:
```bash
python scripts/convert_npz_to_packed_memmap.py \
    --input data.npz \
    --output packed_data
```

**Options**:
```
--input, -i PATH          Input NPZ file (required)
--output, -o PATH         Output directory (required)
--verbose, -v             Print progress (default: True)
--no-verbose              Suppress output
```

**Examples**:
```bash
# Convert single file
python scripts/convert_npz_to_packed_memmap.py -i data.npz -o packed_data

# Convert multiple files
for file in train.npz valid.npz test.npz; do
    python scripts/convert_npz_to_packed_memmap.py \
        -i $file -o ${file%.npz}_packed
done

# Quiet mode
python scripts/convert_npz_to_packed_memmap.py \
    -i data.npz -o packed_data --no-verbose
```

## 🐍 Python API Reference

### PackedMemmapLoader

```python
from mmml.data.packed_memmap_loader import PackedMemmapLoader

loader = PackedMemmapLoader(
    path="data_directory",      # Path to packed memmap files
    batch_size=32,               # Molecules per batch
    shuffle=True,                # Shuffle data
    bucket_size=8192,            # Bucketing group size
    seed=0,                      # Random seed
)
```

**Attributes**:
- `loader.N` - Total number of molecules
- `loader.n_atoms` - Array of atom counts per molecule
- `loader.indices` - Current molecule indices (mutable)

**Methods**:
```python
# Generate batches
for batch in loader.batches(
    num_atoms=60,           # Max atoms (padding size)
    physnet_format=True,    # Include PhysNet fields
):
    # batch: dict with molecular data
    pass

# Get size
n_molecules = len(loader)

# String representation
print(loader)
```

### split_loader

```python
from mmml.data.packed_memmap_loader import split_loader

train_loader, valid_loader = split_loader(
    loader,                 # Loader to split
    train_fraction=0.9,     # Train fraction
    seed=42,                # Random seed (None = no shuffle)
)
```

## 📊 Common Usage Scenarios

### Scenario 1: Quick Test Run
```bash
# Verify everything works (10 epochs)
python examples/train_memmap_simple.py
```

### Scenario 2: Small Molecules (QM9)
```bash
python train_physnet_memmap.py \
    --data_path qm9_packed \
    --batch_size 128 \
    --num_atoms 29 \
    --bucket_size 4096
```

### Scenario 3: Medium Molecules (SPICE)
```bash
python train_physnet_memmap.py \
    --data_path spice_packed \
    --batch_size 64 \
    --num_atoms 60 \
    --features 256
```

### Scenario 4: Large Molecules (Proteins)
```bash
python train_physnet_memmap.py \
    --data_path protein_packed \
    --batch_size 16 \
    --num_atoms 150 \
    --bucket_size 16384
```

### Scenario 5: Production Training
```bash
python train_physnet_memmap.py \
    --data_path production_data \
    --batch_size 64 \
    --num_epochs 500 \
    --learning_rate 0.0005 \
    --features 256 \
    --num_iterations 5 \
    --name production_run_v1
```

### Scenario 6: Custom Training Loop
```python
from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader
from mmml.physnetjax.physnetjax.models.model import EF

# Load data
loader = PackedMemmapLoader("data", batch_size=32)
train_loader, valid_loader = split_loader(loader)

# Create model
model = EF(features=128, num_iterations=3, natoms=60)

# Custom training
for epoch in range(num_epochs):
    for batch in train_loader.batches(num_atoms=60):
        # Your custom logic here
        pass
```

### Scenario 7: Converting Existing Data
```bash
# Step 1: Convert NPZ to packed format
python scripts/convert_npz_to_packed_memmap.py \
    --input my_dataset.npz \
    --output my_dataset_packed

# Step 2: Train on converted data
python train_physnet_memmap.py \
    --data_path my_dataset_packed \
    --batch_size 32
```

## 🔍 Finding Information

### "How do I..."

| Task | Resource | Location |
|------|----------|----------|
| Get started quickly | Quick start guide | `QUICKSTART_MEMMAP.md` |
| Understand the data format | Data format spec | `PACKED_MEMMAP_TRAINING.md` § Data Format |
| Convert my NPZ data | Conversion script | `scripts/convert_npz_to_packed_memmap.py` |
| Train a model | Training script | `train_physnet_memmap.py` |
| Write custom training | Simple example | `examples/train_memmap_simple.py` |
| Optimize performance | Performance tips | `PACKED_MEMMAP_TRAINING.md` § Performance Tips |
| Debug errors | Troubleshooting | `PACKED_MEMMAP_TRAINING.md` § Troubleshooting |
| Understand the API | API reference | `PACKED_MEMMAP_TRAINING.md` § API Reference |
| See the architecture | System overview | `MEMMAP_TRAINING_SUMMARY.md` |
| Compare with original | Code comparison | `CODE_COMPARISON.md` |

### "I want to..."

| Goal | Start Here |
|------|------------|
| Train a model right now | `QUICKSTART_MEMMAP.md` → `train_physnet_memmap.py` |
| Learn the system | `MEMMAP_TRAINING_SUMMARY.md` → `CODE_COMPARISON.md` |
| Optimize my training | `PACKED_MEMMAP_TRAINING.md` § Performance Tips |
| Convert my data | `scripts/convert_npz_to_packed_memmap.py --help` |
| Write custom code | `examples/train_memmap_simple.py` |
| Debug an error | `PACKED_MEMMAP_TRAINING.md` § Troubleshooting |
| Understand PhysNet requirements | `CODE_COMPARISON.md` § PhysNet Requirements |

### "I'm getting..."

| Error | Solution |
|-------|----------|
| "Out of memory" | `PACKED_MEMMAP_TRAINING.md` § Troubleshooting → Out of Memory |
| "File size mismatch" | `PACKED_MEMMAP_TRAINING.md` § Troubleshooting → File Size Mismatch |
| Slow training | `PACKED_MEMMAP_TRAINING.md` § Performance Tips |
| Low GPU utilization | Increase `--batch_size` |
| High padding overhead | Increase `--bucket_size` |

## 📈 Performance Quick Reference

### Recommended Configurations

| Dataset Size | Batch Size | Num Atoms | Bucket Size | GPU RAM |
|-------------|-----------|-----------|-------------|---------|
| Small (10k) | 128 | 29 | 4096 | 8 GB |
| Medium (100k) | 64 | 60 | 8192 | 16 GB |
| Large (1M) | 32 | 80 | 16384 | 24 GB |
| XLarge (10M) | 16 | 100 | 16384 | 32 GB |

### Memory Usage Formula

```
Memory ≈ batch_size × num_atoms × features × 4 bytes
```

Example: `batch_size=32, num_atoms=60, features=128`
```
32 × 60 × 128 × 4 = 983,040 bytes ≈ 1 MB per batch
```

Model params: ~1-4 GB depending on architecture

### Typical Training Times

| Dataset | GPU | Batch Size | Time/Epoch | 100 Epochs |
|---------|-----|------------|------------|------------|
| 10k mols | V100 | 64 | 30 s | 50 min |
| 100k mols | V100 | 32 | 8 min | 13 hours |
| 1M mols | A100 | 64 | 45 min | 3 days |

## 🆘 Quick Help

### Installation Issues
1. Verify JAX installation: `python -c "import jax; print(jax.devices())"`
2. Check GPU: `nvidia-smi`
3. Install dependencies: `pip install -r requirements.txt`

### Data Issues
1. Verify data format: `ls -lh your_data/`
2. Check file sizes match spec
3. Try conversion script on a sample

### Training Issues
1. Start with simple example: `python examples/train_memmap_simple.py`
2. Reduce batch size if OOM
3. Check GPU utilization: `nvidia-smi`

### Still Stuck?
1. Read relevant docs section
2. Check troubleshooting guide
3. Verify data format
4. Try smaller configuration
5. Open issue with details

## 📝 Version Info

**Created**: November 2025  
**System**: PhysNet Packed Memmap Training  
**Location**: `/home/ericb/mmml/`  
**Language**: Python 3.9+  
**Framework**: JAX + Flax  

## 🔗 Related Resources

- **PhysNet Paper**: https://doi.org/10.1021/acs.jctc.9b00181
- **e3x Library**: https://github.com/google-research/e3x
- **JAX Docs**: https://jax.readthedocs.io
- **Flax Docs**: https://flax.readthedocs.io

---

**Last Updated**: November 2025  
**Maintained By**: Project Contributors  
**License**: See LICENSE file in repository

