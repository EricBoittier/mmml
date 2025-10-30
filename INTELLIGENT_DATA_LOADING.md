# Intelligent Data Loading and Field Detection

## 🎯 Overview

The DCMNet data loader now features **intelligent field detection** that automatically determines what each field represents based on its shape and size. This creates harmony between diverse dataset formats.

## 🧠 Smart Field Detection

### 1. D Field (Dipole)

The loader detects three possible meanings based on shape:

| Size | Interpretation | Shape | Example |
|------|----------------|-------|---------|
| `n_samples × 1` | Dipole magnitude | `(1983, 1)` | `\|\|μ\|\|` |
| `n_samples × 3` | Molecular dipole vector | `(1983, 3)` | `[μx, μy, μz]` |
| `n_samples × natoms × 3` | Atom-centered dipoles | `(1983, 18, 3)` | Per-atom dipoles |

**Example Detection:**
```python
# Your data has D with shape (1983, 3)
# Output: "D field: molecular dipole vector (shape: (1983, 3))"

# If D had shape (1983,)
# Output: "D field: dipole magnitude (shape: (1983, 1))"

# If D had shape (1983, 18, 3)
# Output: "D field: atom-centered dipoles (shape: (1983, 18, 3))"
```

### 2. Q Field (Charge/Quadrupole)

The loader detects four possible meanings:

| Size | Interpretation | Shape | Example |
|------|----------------|-------|---------|
| `n_samples × 1` | Total molecular charge | `(1983, 1)` | Net charge |
| `n_samples × 6` | Quadrupole moment | `(1983, 6)` | `[Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]` |
| `n_samples × 9` | Quadrupole tensor | `(1983, 3, 3)` | Full 3×3 tensor |
| `n_samples × natoms` | Atomic charges | `(1983, 18)` | Per-atom charges (should be 'mono') |

**Example Detection:**
```python
# Your data has Q with size 17847 = 1983 × 9
# Output: "Q field: quadrupole tensor (3×3, shape: (1983, 3, 3))"

# If Q had size 1983
# Output: "Q field: total molecular charge (shape: (1983, 1))"

# If Q had size 11898 = 1983 × 6
# Output: "Q field: quadrupole moment (6 components, shape: (1983, 6))"

# If Q had size 35694 = 1983 × 18
# Output: "⚠️  Q field: appears to be atomic charges (shape: (1983, 18))"
#         "   If this is correct, consider renaming to 'mono' in your dataset"
```

### 3. Dxyz Field (Molecular Dipole Vector)

Expected to be molecular dipole in Cartesian coordinates:

| Expected Shape | Interpretation |
|----------------|----------------|
| `(n_samples, 3)` | Molecular dipole vector |

**Validation:**
- Automatically reshapes if flattened
- Warns if unexpected shape
- Ensures proper (n, 3) format

### 4. mono Field (Atomic Charges/Monopoles)

| Status | Action |
|--------|--------|
| Present | Use as-is `(n_samples, natoms)` |
| Missing | Create zeros `(n_samples, natoms)` |

**Note:** If your dataset has atomic charges in `Q`, rename it to `mono` for clarity!

## 📊 Detection Logic Flow

```
Load Field → Check Dimensionality → Infer Meaning → Reshape → Report
    ↓              ↓                      ↓             ↓         ↓
  Field         1D/2D/3D           Based on size    To std    Print info
```

### Example: D Field Detection

```python
if total_size == n_molecules:
    # Scalar per molecule → magnitude
    dataD = dataD.reshape(-1, 1)
    print("D field: dipole magnitude")
    
elif total_size == n_molecules * 3:
    # 3 values per molecule → vector
    dataD = dataD.reshape(-1, 3)
    print("D field: molecular dipole vector")
    
elif total_size == n_molecules * natoms * 3:
    # 3 values per atom → atom-centered
    dataD = dataD.reshape(n_molecules, natoms, 3)
    print("D field: atom-centered dipoles")
```

## 🎓 Semantic Field Types

### Dipole-Related Fields

| Field Name | Typical Shape | Meaning | Usage |
|------------|---------------|---------|-------|
| `D` | `(n, 1)` or `(n, 3)` or `(n, natoms, 3)` | Dipole (various forms) | Auto-detected |
| `Dxyz` | `(n, 3)` | Molecular dipole vector | Validation target |
| `dipole` | `(n, 3)` | Molecular dipole vector | Same as Dxyz |

### Charge/Multipole Fields

| Field Name | Typical Shape | Meaning | Usage |
|------------|---------------|---------|-------|
| `mono` | `(n, natoms)` | Atomic monopole charges | Training target |
| `Q` | `(n, 1)` or `(n, 6)` or `(n, 3, 3)` | Charge/Quadrupole | Auto-detected |

### Grid/Geometry Fields

| Field Name | Typical Shape | Meaning | Usage |
|------------|---------------|---------|-------|
| `vdw_surface` | `(n, n_grid, 3)` | VDW surface grid | ESP evaluation |
| `esp_grid` | `(n, n_grid, 3)` | Same as vdw_surface | Auto-translated |
| `n_grid` | `(n, 1)` | Grid points per molecule | For variable grids |

## 🔍 Output Messages Guide

### Information Messages (✓)
```
   D field: molecular dipole vector (shape: (1983, 3))
   Q field: quadrupole tensor (3×3, shape: (1983, 3, 3))
   Dxyz field: molecular dipole vector (shape: (1983, 3))
```
**Action:** None needed - field correctly detected

### Warning Messages (⚠️)
```
⚠️  No 'mono' field found in dataset
   Creating zero charges for 1983 molecules × 18 atoms
```
**Action:** This is OK if you're training ESP-only without charge constraints

```
⚠️  Q field: appears to be atomic charges (shape: (1983, 18))
   If this is correct, consider renaming to 'mono' in your dataset
```
**Action:** If Q actually contains atomic charges, rename the field to 'mono' in your data file

```
⚠️  D field has unexpected size: 12345
   Expected: 1983 (mag), 5949 (vec), or 107028 (atom)
   Reshaped to: (1983, 6)
```
**Action:** Check your data - field might be corrupted or in an unusual format

## 🎯 Dataset Compatibility Matrix

### Standard Format (QM9-style)
```python
{
    'R': (n, natoms, 3),        # Positions
    'Z': (n, natoms),           # Atomic numbers
    'N': (n, 1),                # Actual atoms per molecule
    'mono': (n, natoms),        # Atomic charges ✅
    'esp': (n, n_grid),         # ESP values
    'vdw_surface': (n, n_grid, 3),  # Grid points ✅
    'n_grid': (n, 1),           # Grid count
    'Dxyz': (n, 3),             # Molecular dipole ✅
}
```

### Your Format (MOLPRO-style)
```python
{
    'R': (n, natoms, 3),        # Positions
    'Z': (n, natoms),           # Atomic numbers  
    'N': (n, 1),                # Actual atoms per molecule
    # 'mono': missing           # ✅ Auto-created as zeros
    'esp': (n, n_grid),         # ESP values
    'esp_grid': (n, n_grid, 3), # ✅ Auto-mapped to 'vdw_surface'
    # 'n_grid': missing         # ✅ Auto-computed
    'D': (n, 3),                # ✅ Auto-detected as molecular dipole
    'Q': (n, 9),                # ✅ Auto-detected as quadrupole tensor
}
```

### Alternative Format
```python
{
    'R': (n, natoms, 3),
    'Z': (n, natoms),
    'mono': (n, natoms),        # Atomic charges
    'esp': (n, n_grid),
    'esp_grid': (n, n_grid, 3), # ✅ Auto-mapped
    'D': (n, 1),                # ✅ Auto-detected as magnitude
    'Q': (n, 1),                # ✅ Auto-detected as total charge
}
```

## 💡 Best Practices

### For Data Providers

1. **Use standard field names** when possible:
   - `mono` for atomic charges (not `Q`)
   - `Dxyz` for molecular dipole vector
   - `vdw_surface` for ESP grid points

2. **Document field meanings** in dataset metadata

3. **Use consistent shapes**:
   - Molecular properties: `(n_samples, ...)`
   - Atomic properties: `(n_samples, natoms, ...)`

### For Data Users

1. **Check the detection messages** when loading:
   ```python
   train_data, valid_data = prepare_datasets(...)
   # Read the output - it tells you what was detected!
   ```

2. **Verify fields are correct**:
   ```python
   print(f"Keys: {train_data.keys()}")
   print(f"Shapes: {[(k, v.shape) for k, v in train_data.items()]}")
   ```

3. **Understand warnings**:
   - ⚠️ warnings indicate detected issues
   - Read the suggestions and check your data

## 🔬 Technical Details

### Shape Inference Algorithm

```python
def detect_field_type(field_data, n_molecules, natoms):
    total_size = field_data.size
    
    # Try different interpretations
    if total_size == n_molecules:
        return "scalar_per_molecule", (n_molecules, 1)
    elif total_size == n_molecules * 3:
        return "vector_per_molecule", (n_molecules, 3)
    elif total_size == n_molecules * 6:
        return "symmetric_tensor_6", (n_molecules, 6)
    elif total_size == n_molecules * 9:
        return "tensor_3x3", (n_molecules, 3, 3)
    elif total_size == n_molecules * natoms:
        return "scalar_per_atom", (n_molecules, natoms)
    elif total_size == n_molecules * natoms * 3:
        return "vector_per_atom", (n_molecules, natoms, 3)
    else:
        return "unknown", (n_molecules, -1)
```

### Supported Field Combinations

The loader now handles **any combination** of these fields with intelligent detection:

✅ Old datasets (QM9, MD17, etc.)  
✅ Quantum chemistry outputs (MOLPRO, Gaussian, etc.)  
✅ Custom datasets with non-standard naming  
✅ Partial datasets (missing fields auto-filled)  
✅ Mixed formats in training pipeline  

## 🎵 Creating Harmony

**Philosophy**: The data loader should understand your data, not force you to understand its format.

**Implementation**:
1. **Detect** - Infer meaning from shape
2. **Normalize** - Convert to standard internal format
3. **Report** - Tell user what was detected
4. **Warn** - Alert on ambiguities
5. **Proceed** - Use best interpretation

## 📝 Example Session Output

```
Loading dataset: water_molecules.npz
shape (1983, 18, 3)
⚠️  No 'mono' field found in dataset
   Creating zero charges for 1983 molecules × 18 atoms
   Actual atoms per molecule will be respected (from 'N' field)
   Note: Q field (if present) is quadrupole, not monopole charges
   D field: molecular dipole vector (shape: (1983, 3))
   Q field: quadrupole tensor (3×3, shape: (1983, 3, 3))
   Dxyz field: molecular dipole vector (shape: (1983, 3))
   
✅ Dataset loaded successfully
   - 1200 training samples
   - 100 validation samples
   - Fields: R, Z, N, mono (zeros), esp, vdw_surface, n_grid, D, Q, Dxyz
```

## 🚀 Benefits

### Flexibility
✅ Works with any reasonable dataset format  
✅ Handles missing fields gracefully  
✅ Auto-detects field semantics from shape  

### Robustness
✅ Clear reporting of what was detected  
✅ Warnings for ambiguous cases  
✅ Fallback strategies for unexpected formats  

### User Experience
✅ No manual preprocessing needed  
✅ Informative messages  
✅ "Just works" with diverse data sources  

## 🎓 Field Semantic Reference

### Dipole Fields

| Field | If Size = n | If Size = n×3 | If Size = n×natoms×3 |
|-------|------------|---------------|----------------------|
| **D** | Magnitude | **Vector** | Atom-centered |
| **Dxyz** | Error | **Vector** | Error |
| **dipole** | Error | **Vector** | Error |

### Charge/Multipole Fields

| Field | If Size = n | If Size = n×6 | If Size = n×9 | If Size = n×natoms |
|-------|------------|---------------|---------------|-------------------|
| **Q** | Total charge | Quadrupole moment | **Quadrupole tensor** | Atomic charges⚠️ |
| **mono** | Error | Error | Error | **Atomic charges** |

### Geometry Fields

| Field | Expected Shape | Alternatives |
|-------|---------------|--------------|
| **vdw_surface** | `(n, n_grid, 3)` | `esp_grid` → auto-mapped |
| **esp** | `(n, n_grid)` | Always required |
| **n_grid** | `(n, 1)` | Auto-computed if missing |
| **R** | `(n, natoms, 3)` | Atomic positions |
| **Z** | `(n, natoms)` | Atomic numbers |
| **N** | `(n, 1)` | Actual atoms (for padding) |

## 🔧 Implementation Example

Here's how the smart detection works in code:

```python
# D Field Detection
if "D" in datasets[0].keys():
    dataD = load_and_concat(dataset["D"])
    
    total_size = dataD.size
    n_molecules = shape[0]
    
    if total_size == n_molecules:
        # Scalar → magnitude
        dataD = dataD.reshape(-1, 1)
        print("D field: dipole magnitude")
    elif total_size == n_molecules * 3:
        # Vector → molecular dipole
        dataD = dataD.reshape(-1, 3)
        print("D field: molecular dipole vector")
    elif total_size == n_molecules * natoms * 3:
        # 3D → atom-centered
        dataD = dataD.reshape(n_molecules, natoms, 3)
        print("D field: atom-centered dipoles")
    else:
        # Unexpected → warn and reshape
        print(f"⚠️  D field unexpected size: {total_size}")
        dataD = dataD.reshape(n_molecules, -1)
    
    data.append(dataD)
    keys.append("D")
```

## 🎯 For Your Specific Case

Your dataset:
```python
Data keys: ['R', 'D', 'Q', 'Z', 'esp', 'esp_grid', 'F']
```

**Detection results:**
- ✅ `R` - positions (standard)
- ✅ `Z` - atomic numbers (standard)
- ✅ `esp` - ESP values (standard)
- ✅ `esp_grid` → `vdw_surface` (auto-mapped)
- ✅ `F` - forces (standard)
- ✅ `D` - will be auto-detected (likely molecular dipole vector if size = n×3)
- ✅ `Q` - will be auto-detected (likely quadrupole tensor if size = n×9)
- ✅ `mono` - will be created as zeros (with warning)
- ✅ `n_grid` - will be auto-computed from ESP shape

## 🚀 Usage

**Restart your Jupyter kernel**, then:

```python
import importlib
from mmml.dcmnet.dcmnet import data
importlib.reload(data)
from mmml.dcmnet.dcmnet.data import prepare_datasets
import jax

key = jax.random.PRNGKey(42)

train_data, valid_data = prepare_datasets(
    key, 
    num_train=1200,
    num_valid=100,
    filename=[data_path],
    natoms=18,
)

# You'll see detailed detection output:
# shape (1983, 18, 3)
# ⚠️  No 'mono' field found in dataset
#    Creating zero charges for 1983 molecules × 18 atoms
#    Actual atoms per molecule will be respected (from 'N' field)
#    Note: Q field (if present) is quadrupole, not monopole charges
#    D field: molecular dipole vector (shape: (1983, 3))
#    Q field: quadrupole tensor (3×3, shape: (1983, 3, 3))
```

## 🎵 Harmony Achieved

The system now:
- ✅ Understands diverse data formats
- ✅ Auto-detects field semantics
- ✅ Provides clear feedback
- ✅ Handles edge cases gracefully
- ✅ Maintains internal consistency
- ✅ Works with partial datasets

**True data harmony through intelligent detection!** 🎵✨

---

**Pro Tip**: Always check the detection messages when loading a new dataset - they tell you exactly how each field was interpreted!

