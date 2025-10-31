# Fixes Applied

## Issue 1: NPZ File with Pickled Data
**Error:** `ValueError: This file contains pickled (object) data`

**Fix:** Changed `allow_pickle=False` to `allow_pickle=True` in `/home/ericb/mmml/mmml/data/loaders.py` (line 109)

**Reason:** The NPZ files contain metadata stored as object dtype arrays, which require pickle support. This is safe when loading trusted data files.

---

## Issue 2: Dipole Array Reshape Error
**Error:** `ValueError: cannot reshape array of size 180 into shape (3,3)`

**Root Cause:** Dipole data was stored as `(n_structures, n_atoms, 3)` (padded per-atom) but the batching code expected `(n_structures, 3)` (per-structure molecular dipoles).

**Fixes Applied:**

### 1. Enhanced `ensure_standard_keys()` function
**File:** `/home/ericb/mmml/examples/co2/physnet_train/trainer.py`

Now properly handles dipoles in multiple formats:
- **2D arrays (n_structures, 3)**: Already correct, kept as-is
- **3D arrays (n_structures, n_atoms, 3)**: Sums over atoms to get molecular dipole
- **1D arrays**: Reshapes to (n_structures, 3)
- Checks both `D` and `Dxyz` keys

### 2. Added `resize_padded_arrays()` function
**File:** `/home/ericb/mmml/examples/co2/physnet_train/trainer.py`

New function that safely resizes pre-padded arrays to match `--natoms`:
- Can expand (add more padding) or truncate (remove excess padding)
- Safety check: Prevents truncation below the maximum actual atom count
- Handles: `R`, `F`, `Z`, `mono` arrays
- Verbose output shows shape transformations

### 3. Integrated resizing into data pipeline
**File:** `/home/ericb/mmml/examples/co2/physnet_train/trainer.py`

Added calls after `ensure_standard_keys()`:
```python
train_data = resize_padded_arrays(train_data, args.natoms, verbose=args.verbose)
valid_data = resize_padded_arrays(valid_data, args.natoms, verbose=args.verbose)
```

### 4. Enhanced diagnostics
Added shape information to data loading output:
- Shows shapes of all major arrays (R, Z, E, F, D)
- Reports maximum atom count in the dataset
- Helps users verify data is correctly formatted

---

## Usage

Now you can safely:

1. **Load NPZ files with metadata** (object arrays)
2. **Use pre-padded data** with any `--natoms` value (as long as it's >= max actual atoms)
3. **Train with dipoles** regardless of how they're stored in the NPZ file

### Example:
```bash
python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --natoms 60 \  # Can differ from data's current padding
  --batch-size 32 \
  --epochs 100 \
  --verbose
```

### Output will show:
```
📏 Resizing arrays from 80 to 60 atoms...
  R: (1000, 80, 3) → (1000, 60, 3)
  F: (1000, 80, 3) → (1000, 60, 3)
  Z: (1000, 80) → (1000, 60)

✅ Data loaded:
  Training samples: 1000
  Array shapes:
    R: (1000, 60, 3)
    Z: (1000, 60)
    E: (1000,)
    F: (1000, 60, 3)
    D: (1000, 3)  # ← Now always correct shape!
  Max atoms in data: 30
```

---

## Safety Features

1. **Resize validation**: Cannot truncate below the maximum atom count in your data
2. **Shape detection**: Automatically detects and fixes incorrectly shaped dipoles
3. **Verbose feedback**: Shows exactly what transformations are applied
4. **Metadata handling**: Properly handles pickled metadata in NPZ files

---

## Technical Details

### Why the error occurred:
When dipoles were stored as `(n_structures=60, n_atoms=60, 3)`:
- Indexing with a batch of 3 structures: `data["D"][perm]` → shape `(3, 60, 3)` = 180 elements
- Batching code tried to reshape to `(batch_size, 3)` = `(3, 3)` = 9 elements
- Result: `ValueError: cannot reshape array of size 180 into shape (3,3)`

### How it's fixed:
1. `ensure_standard_keys()` detects the 3D dipole array
2. Sums over the atom dimension: `(3, 60, 3)` → `(3, 3)` via `sum(axis=1)`
3. This gives the molecular dipole (total dipole for each structure)
4. Batching code now receives the expected shape

---

## Files Modified

1. `/home/ericb/mmml/mmml/data/loaders.py` - Enable pickle loading
2. `/home/ericb/mmml/examples/co2/physnet_train/trainer.py` - Multiple improvements:
   - Enhanced `ensure_standard_keys()`
   - Added `resize_padded_arrays()`
   - Integrated resizing into pipeline
   - Enhanced diagnostics

All changes maintain backward compatibility!

