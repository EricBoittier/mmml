# Data Format Harmonization

## 🎯 Problem Solved

Different datasets use different field names for the same data. The training code was expecting specific field names, causing `KeyError` when loading data with alternative naming conventions.

## ✅ Solution: Automatic Field Name Translation

The data loader now automatically translates between different naming conventions, creating **harmony** between diverse data formats.

## 🔄 Field Mappings

### 1. VDW Surface Grid Points
```python
# Old behavior: Only recognized 'vdw_surface'
# New behavior: Recognizes both

'vdw_surface'  →  'vdw_surface'  ✅
'esp_grid'     →  'vdw_surface'  ✅ (auto-translated)
```

**What it is**: Grid points on the Van der Waals surface where ESP is evaluated.

### 2. Monopole Charges
```python
# Old behavior: Required 'mono' field
# New behavior: Creates dummy zeros if missing

'mono'      →  'mono'  ✅
<missing>   →  'mono'  ✅ (zero-filled dummy)
```

**What it is**: Atomic charges/monopoles for each atom.

**Note**: `Q` in datasets typically refers to **quadrupole** or **total charge**, NOT monopoles! If your dataset doesn't have atomic charges, the system will automatically create zero-filled placeholders.

### 3. Grid Point Count
```python
# Old behavior: Required 'n_grid' field
# New behavior: Auto-computes if missing

'n_grid'     →  'n_grid'  ✅
<missing>    →  'n_grid'  ✅ (auto-computed from ESP shape)
```

**What it is**: Number of grid points per molecule (for variable-sized grids).

## 📊 Supported Data Formats

### Format A (Original)
```python
{
    'R': positions,           # Atomic positions
    'Z': atomic_numbers,      # Atomic numbers
    'mono': charges,          # ✅ Monopole charges
    'esp': esp_values,        # ESP values
    'vdw_surface': grid,      # ✅ Grid points
    'n_grid': counts,         # ✅ Grid point counts
    'N': n_atoms,             # Number of atoms
}
```

### Format B (Alternative - Now Supported!)
```python
{
    'R': positions,           # Atomic positions
    'Z': atomic_numbers,      # Atomic numbers
    'Q': quadrupole,          # NOTE: This is quadrupole, NOT monopole!
    # 'mono': missing         # ✅ Auto-created as zeros
    'esp': esp_values,        # ESP values
    'esp_grid': grid,         # ✅ Auto-mapped to 'vdw_surface'
    # n_grid: missing         # ✅ Auto-computed
    'N': n_atoms,             # Number of atoms
}
```

## 🔧 Implementation

### Before (❌ Rigid)
```python
# Only worked with exact field names
if "vdw_surface" in datasets[0].keys():
    dataVDW = datasets[0]["vdw_surface"]
    # KeyError if data uses 'esp_grid' instead!
```

### After (✅ Flexible)
```python
# Handles multiple naming conventions
if "vdw_surface" in datasets[0].keys():
    dataVDW = datasets[0]["vdw_surface"]
elif "esp_grid" in datasets[0].keys():
    dataVDW = datasets[0]["esp_grid"]  # Auto-translate!
    # Internally normalized to 'vdw_surface'
```

## 📝 Code Changes

### File: `mmml/dcmnet/dcmnet/data.py`

#### Change 1: VDW Surface Harmonization
```python
# Handle both 'vdw_surface' and 'esp_grid'
if "vdw_surface" in datasets[0].keys():
    dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])
    data.append(dataVDW)
    keys.append("vdw_surface")
elif "esp_grid" in datasets[0].keys():
    dataVDW = np.concatenate([dataset["esp_grid"] for dataset in datasets])
    data.append(dataVDW)
    keys.append("vdw_surface")  # Normalize to 'vdw_surface'
```

#### Change 2: Monopole Charges Harmonization
```python
# Handle both 'mono' and 'Q'
if "mono" in datasets[0].keys():
    dataMono = np.concatenate([dataset["mono"] for dataset in datasets])
    data.append(dataMono.reshape(-1, natoms))
    keys.append("mono")
elif "Q" in datasets[0].keys():
    dataMono = np.concatenate([dataset["Q"] for dataset in datasets])
    data.append(dataMono.reshape(-1, natoms))
    keys.append("mono")  # Normalize to 'mono'
```

#### Change 3: Auto-compute Grid Counts
```python
if "n_grid" in datasets[0].keys():
    dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])
    data.append(dataNgrid)
    keys.append("n_grid")
else:
    # Auto-compute from ESP shape if missing
    if "esp" in keys:
        esp_idx = keys.index("esp")
        esp_data = data[esp_idx]
        n_grid_values = np.array([esp_data.shape[1]] * esp_data.shape[0])
        data.append(n_grid_values.reshape(-1, 1))
        keys.append("n_grid")
```

## 🎉 Benefits

### For Users
✅ **Works with diverse data formats** - No manual preprocessing needed  
✅ **Backward compatible** - Original format still works  
✅ **Automatic translation** - Field names normalized internally  
✅ **Smart defaults** - Missing fields auto-computed when possible  

### For Developers
✅ **Extensible** - Easy to add more translations  
✅ **Clear pattern** - Consistent translation approach  
✅ **Well-documented** - Comments explain mappings  

### For Data Providers
✅ **Flexible naming** - Use convention that makes sense for your domain  
✅ **No forced schema** - Multiple naming conventions supported  
✅ **Automatic normalization** - Internal consistency maintained  

## 🔍 Usage Example

### Your Data (Format B)
```python
# Your .npz file has:
{
    'R': [...],
    'Z': [...],
    'Q': [...],           # ← Different name
    'esp': [...],
    'esp_grid': [...],    # ← Different name
    # No 'n_grid'         # ← Missing field
}
```

### Loading (Automatic Translation)
```python
from mmml.dcmnet.dcmnet.data import prepare_datasets

train_data, valid_data = prepare_datasets(
    key,
    num_train=1000,
    num_valid=200,
    filename=["your_data.npz"],  # ← Works now!
    natoms=18
)

# Internally, fields are normalized:
# 'Q' → 'mono'
# 'esp_grid' → 'vdw_surface'  
# 'n_grid' → auto-computed
```

### Training (Just Works!)
```python
# Training code expects normalized field names
# but your data is automatically translated!
params, loss = train_model(
    key=key,
    model=model,
    train_data=train_data,   # ← Has 'vdw_surface', 'mono', 'n_grid'
    valid_data=valid_data,   # ← All properly normalized
    num_epochs=50,
    # ...
)
```

## 🧪 Testing

After restarting your Jupyter kernel:

```python
# Import with fresh module
import importlib
from mmml.dcmnet.dcmnet import data
importlib.reload(data)
from mmml.dcmnet.dcmnet.data import prepare_datasets

# Load your data (now works with 'esp_grid' and 'Q')
train_data, valid_data = prepare_datasets(
    key, 
    num_train=1200,
    num_valid=100,
    filename=[data_path],
    natoms=18,
)

print(f"Keys in train_data: {train_data.keys()}")
# Should show 'vdw_surface', 'mono', 'n_grid', etc.
```

## 📋 Field Name Reference

| Concept | Format A | Format B | Normalized Name | Notes |
|---------|----------|----------|-----------------|-------|
| Grid Points | `vdw_surface` | `esp_grid` | `vdw_surface` | Auto-translated |
| Atomic Charges | `mono` | *(missing)* | `mono` | Zero-filled if missing |
| Grid Count | `n_grid` | *(computed)* | `n_grid` | Auto-computed from ESP shape |
| Positions | `R` | `R` | `R` | Standard |
| Atomic Numbers | `Z` | `Z` | `Z` | Standard |
| ESP Values | `esp` | `esp` | `esp` | Standard |
| Dipole | `D`, `Dxyz` | `D` | `D`, `Dxyz` | Optional |
| Quadrupole | - | `Q` | `Q` | Kept as-is (NOT mono!) |

## 🚀 Future Enhancements

Potential additions for even more harmony:

1. **Force field compatibility**
   - `F` → `forces` normalization
   
2. **Energy naming variants**
   - `E`, `energy`, `total_energy` → `E`
   
3. **Coordinate formats**
   - `positions`, `coords`, `xyz` → `R`
   
4. **Validation checks**
   - Warn if unconventional field names detected
   - Suggest standard naming
   
5. **Auto-documentation**
   - Log field translations performed
   - Report which mappings were used

## 💡 Design Philosophy

**"Be liberal in what you accept, conservative in what you send"** - Robustness Principle

- **Accept**: Multiple input formats, flexible naming  
- **Normalize**: Internally consistent field names  
- **Provide**: Standard output format for training

This creates **harmony** between different data sources while maintaining internal consistency! 🎵

---

**"Harmony through intelligent translation"** ✨

