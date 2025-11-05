# Packed Memory-Mapped Data Training for PhysNet

This guide explains how to train PhysNet models using the `PackedMemmapLoader`, which provides efficient loading of large molecular datasets stored in memory-mapped format.

## Overview

The `PackedMemmapLoader` is designed for training on large-scale molecular property prediction datasets (like OpenQDC) that don't fit entirely in memory. It uses:

- **Memory mapping**: Data stays on disk, only loaded when needed
- **Packed storage**: Variable-size molecules stored efficiently without wasted space
- **Bucketed batching**: Molecules sorted by size to minimize padding overhead
- **PhysNet compatibility**: Automatic conversion to PhysNet's expected format

## Data Format

### Required Files

Your data directory should contain:

```
data_path/
├── offsets.npy       # (N+1,) int64 - Cumulative atom offsets
├── n_atoms.npy       # (N,) int32 - Number of atoms per molecule  
├── Z_pack.int32      # (sum_atoms,) int32 - Packed atomic numbers
├── R_pack.f32        # (sum_atoms, 3) float32 - Packed positions (Angstrom)
├── F_pack.f32        # (sum_atoms, 3) float32 - Packed forces (kcal/mol/Å)
├── E.f64             # (N,) float64 - Energies (kcal/mol)
└── Qtot.f64          # (N,) float64 - Total charges (optional)
```

### Data Structure

The "packed" format stores all molecules concatenated:
- Molecule `i` has atoms from `offsets[i]` to `offsets[i+1]`
- `offsets[-1]` equals the total number of atoms across all molecules
- This eliminates wasted space from padding small molecules

### Example: Creating Packed Data

```python
import numpy as np

# Example: 3 molecules with different sizes
molecules = [
    {"Z": [6, 1, 1, 1, 1], "R": ...},      # CH4 (5 atoms)
    {"Z": [8, 1, 1], "R": ...},            # H2O (3 atoms)  
    {"Z": [6, 6, 1, 1, 1, 1], "R": ...},   # C2H4 (6 atoms)
]

# Pack into arrays
all_Z = []
all_R = []
offsets = [0]

for mol in molecules:
    all_Z.extend(mol["Z"])
    all_R.extend(mol["R"])
    offsets.append(len(all_Z))

Z_pack = np.array(all_Z, dtype=np.int32)
R_pack = np.array(all_R, dtype=np.float32)
offsets = np.array(offsets, dtype=np.int64)
n_atoms = np.array([len(m["Z"]) for m in molecules], dtype=np.int32)

# Save as memory-mapped files
Z_pack.tofile("data_path/Z_pack.int32")
R_pack.tofile("data_path/R_pack.f32")
np.save("data_path/offsets.npy", offsets)
np.save("data_path/n_atoms.npy", n_atoms)
```

## Usage

### 1. Simple Training Script

```python
from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader
from mmml.physnetjax.physnetjax.models.model import EF

# Load data
loader = PackedMemmapLoader(
    path="openqdc_packed_memmap",
    batch_size=32,
    shuffle=True,
    bucket_size=8192,
    seed=42,
)

# Split into train/validation
train_loader, valid_loader = split_loader(loader, train_fraction=0.9)

# Create model
model = EF(
    features=128,
    num_iterations=3,
    cutoff=5.0,
    natoms=60,  # Max atoms per molecule
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader.batches(num_atoms=60):
        # batch contains: Z, R, F, E, N, Qtot, dst_idx, src_idx, batch_segments
        # Train step...
        pass
```

### 2. Command-Line Training

Use the provided training script:

```bash
python train_physnet_memmap.py \
    --data_path openqdc_packed_memmap \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --num_atoms 60 \
    --features 128 \
    --cutoff 5.0
```

Full options:

```bash
python train_physnet_memmap.py --help
```

**Key arguments:**
- `--data_path`: Path to packed memmap directory (required)
- `--batch_size`: Molecules per batch (default: 32)
- `--num_atoms`: Max atoms to pad to (default: 60)
- `--bucket_size`: Size of bucketing groups (default: 8192)
- `--features`: Hidden layer size (default: 128)
- `--num_iterations`: Message passing iterations (default: 3)
- `--cutoff`: Interaction cutoff in Å (default: 5.0)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_epochs`: Training epochs (default: 100)

### 3. Running the Simple Example

```bash
python examples/train_memmap_simple.py
```

This runs a minimal 10-epoch training loop to verify everything works.

## Advanced Usage

### Custom Bucketing Strategy

Bucket size controls the tradeoff between padding overhead and shuffling:

```python
loader = PackedMemmapLoader(
    path="data",
    batch_size=32,
    bucket_size=4096,  # Smaller = more shuffle, more padding
                       # Larger = less shuffle, less padding
)
```

**Guidelines:**
- Small molecules (< 20 atoms): `bucket_size=4096`
- Medium molecules (20-50 atoms): `bucket_size=8192` (default)
- Large molecules (> 50 atoms): `bucket_size=16384`

### Non-PhysNet Format

If you want simple padded batches without PhysNet graph indices:

```python
for batch in loader.batches(num_atoms=60, physnet_format=False):
    # batch contains: Z, R, F, E, mask, Qtot
    # mask is (B, Amax) boolean array indicating real atoms
    Z = batch["Z"]        # (B, Amax)
    mask = batch["mask"]  # (B, Amax) 
    
    # Use mask to ignore padding
    real_atoms = Z[mask]
```

### Manual Train/Validation Split

```python
loader = PackedMemmapLoader(path="data", batch_size=32)

# Manually set indices for train split
train_loader = PackedMemmapLoader(path="data", batch_size=32)
train_loader.indices = loader.indices[:9000]
train_loader.N = 9000

# Validation split
valid_loader = PackedMemmapLoader(path="data", batch_size=32, shuffle=False)
valid_loader.indices = loader.indices[9000:]
valid_loader.N = len(loader) - 9000
```

## Performance Tips

### 1. Choose Appropriate Batch Size

Memory usage scales with `batch_size * num_atoms`:

- **GPU with 8GB**: `batch_size=32`, `num_atoms=60`
- **GPU with 16GB**: `batch_size=64`, `num_atoms=80`  
- **GPU with 24GB**: `batch_size=128`, `num_atoms=100`

### 2. Optimize Bucket Size

Monitor padding overhead:

```python
for batch in loader.batches(num_atoms=60):
    actual_atoms = batch["N"].sum()
    padded_atoms = batch["Z"].size
    efficiency = actual_atoms / padded_atoms
    print(f"Batch efficiency: {efficiency:.1%}")
```

Aim for > 70% efficiency. If lower, increase `bucket_size`.

### 3. Preload Validation Data

Validation batches don't change, so preload them:

```python
valid_batches = list(valid_loader.batches(num_atoms=60))

for epoch in range(num_epochs):
    # Train
    for batch in train_loader.batches(num_atoms=60):
        train_step(batch)
    
    # Validate (no I/O overhead)
    for batch in valid_batches:
        eval_step(batch)
```

### 4. Use SSD Storage

Memory-mapped files benefit greatly from fast storage:
- **HDD**: ~100 MB/s → bottleneck for large batches
- **SSD**: ~500 MB/s → adequate for most use cases
- **NVMe**: ~3000 MB/s → no I/O bottleneck

## Troubleshooting

### Out of Memory Errors

**Symptom:** `RuntimeError: RESOURCE_EXHAUSTED: Out of memory`

**Solutions:**
1. Reduce `batch_size`
2. Reduce `num_atoms` 
3. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` environment variable
4. Use smaller model (`--features 64` instead of `128`)

### Slow Training

**Symptom:** Long time per epoch, low GPU utilization

**Solutions:**
1. Increase `batch_size` to saturate GPU
2. Verify data is on fast storage (SSD/NVMe)
3. Preload validation batches (see Performance Tips)
4. Ensure `shuffle=True` for training loader

### File Size Mismatch Errors

**Symptom:** `ValueError: Z_pack.int32 size mismatch`

**Solutions:**
1. Verify all files were fully written (no truncation)
2. Check `offsets[-1]` matches total atoms in packed arrays
3. Regenerate packed data files

### Memory Leak During Training

**Symptom:** Memory usage grows over epochs

**Solutions:**
1. Ensure you're not accumulating batches in memory
2. Delete batch after use: `del batch` 
3. Regenerate batches each epoch (don't cache train batches)

## API Reference

### PackedMemmapLoader

```python
class PackedMemmapLoader:
    def __init__(
        self,
        path: str,
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int = 8192,
        seed: int = 0,
    )
```

**Methods:**
- `batches(num_atoms=None, physnet_format=True)`: Generate batches
- `__len__()`: Returns total number of molecules
- `__repr__()`: String representation

**Attributes:**
- `N`: Total molecules in dataset
- `n_atoms`: Array of atom counts per molecule
- `indices`: Current molecule indices (can be modified)

### split_loader

```python
def split_loader(
    loader: PackedMemmapLoader,
    train_fraction: float = 0.9,
    seed: int = None,
) -> tuple[PackedMemmapLoader, PackedMemmapLoader]:
```

Splits a loader into train and validation loaders.

**Returns:** `(train_loader, valid_loader)`

## Examples

### Example 1: Quick Training Run

```bash
# 10 epochs for testing
python train_physnet_memmap.py \
    --data_path my_data \
    --num_epochs 10 \
    --batch_size 16
```

### Example 2: Large Model Training

```bash
# Production training
python train_physnet_memmap.py \
    --data_path openqdc_packed_memmap \
    --num_epochs 500 \
    --batch_size 64 \
    --num_atoms 80 \
    --features 256 \
    --num_iterations 5 \
    --learning_rate 0.0005 \
    --name production_run
```

### Example 3: Small Molecules

```bash
# Optimize for small molecules
python train_physnet_memmap.py \
    --data_path qm9_packed \
    --batch_size 128 \
    --num_atoms 29 \
    --bucket_size 4096 \
    --features 128
```

## Comparison with Standard NPZ Loading

| Feature | NPZ Loading | Packed Memmap |
|---------|-------------|---------------|
| Memory usage | High (loads all data) | Low (streams from disk) |
| Initial load time | Slow (minutes) | Fast (seconds) |
| Batch generation | Fast (in-memory) | Medium (disk I/O) |
| Dataset size limit | RAM size | Disk size |
| Padding overhead | Fixed | Adaptive (bucketing) |
| Best for | Small datasets (< 10GB) | Large datasets (> 10GB) |

## Further Reading

- PhysNet paper: [Unke & Meuwly, 2019](https://doi.org/10.1021/acs.jctc.9b00181)
- OpenQDC dataset: [openqdc documentation](https://github.com/valence-labs/openqdc)
- JAX documentation: [jax.readthedocs.io](https://jax.readthedocs.io)

## Support

For issues or questions:
1. Check this documentation
2. Review `examples/train_memmap_simple.py`
3. Open an issue on GitHub with:
   - Full error message
   - Data format description
   - Command/code used
   - System info (GPU, RAM, storage type)

