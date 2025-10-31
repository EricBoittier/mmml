# Merging Multiwfn Charge Files

## Overview

The `merge_charge_files.py` script combines multiple Multiwfn charge analysis NPZ files into a single dataset for training.

## Usage

### Basic Usage
```bash
# Merge all charge files in a directory
python merge_charge_files.py /path/to/charge/files/ -o merged_charges.npz
```

### From Cluster (Copy files first)
```bash
# 1. Copy charge files from cluster
scp user@cluster:/cluster/home/boittier/carb/jobs/*_charges.npz ./charge_files/

# 2. Merge them
python merge_charge_files.py ./charge_files/ -o co2_charges_all.npz --compress
```

### Filter by Quantum Method
```bash
# Merge only MP2 charges
python merge_charge_files.py ./charge_files/ -o co2_charges_mp2.npz --filter-method mp2

# Merge only HF charges
python merge_charge_files.py ./charge_files/ -o co2_charges_hf.npz --filter-method hf
```

### Advanced Options
```bash
python merge_charge_files.py INPUT_DIR \
  -o output.npz \
  --pattern "*mp2_charges.npz" \
  --compress \
  --filter-method mp2
```

## Output Format

The merged NPZ file contains:

### Charge Methods (all in atomic units, electron charge)
- `charges_hirshfeld`: (n_samples, n_atoms) Hirshfeld charges
- `charges_vdd`: (n_samples, n_atoms) Voronoi deformation density
- `charges_becke`: (n_samples, n_atoms) Becke charges
- `charges_adch`: (n_samples, n_atoms) ADCH charges
- `charges_chelpg`: (n_samples, n_atoms) CHELPG charges
- `charges_mk`: (n_samples, n_atoms) Merz-Kollman charges
- `charges_cm5`: (n_samples, n_atoms) CM5 charges
- `charges_mbis`: (n_samples, n_atoms) MBIS charges

### MBIS Additional Data
- `mbis_charges_raw`: (n_samples, n_atoms) Raw MBIS charges
- `mbis_dipoles`: (n_samples, n_atoms, 3) Atomic dipoles
- `mbis_quadrupole_cartesian`: (n_samples, n_atoms, 6) Quadrupoles
- `mbis_quadrupole_traceless`: (n_samples, n_atoms, 6) Traceless quadrupoles

### Geometry Metadata
- `atoms`: Element symbols for each sample
- `coordinates_angstrom`: (n_samples, n_atoms, 3) Atomic positions [Angstrom]
- `r1`: (n_samples,) First C-O bond length [Angstrom]
- `r2`: (n_samples,) Second C-O bond length [Angstrom]
- `angle`: (n_samples,) O-C-O angle [degrees]
- `qm_method`: (n_samples,) Quantum method ('hf' or 'mp2')

## Example: Load and Use Merged Data

```python
import numpy as np

# Load merged charges
data = np.load('merged_charges.npz', allow_pickle=True)

# Access different charge methods
hirshfeld = data['charges_hirshfeld']  # (n_samples, n_atoms)
mk = data['charges_mk']  # Merz-Kollman ESP-fitted charges
mbis = data['charges_mbis']  # MBIS charges

# Access geometry info
r1_values = data['r1']
r2_values = data['r2']
angles = data['angle']

# Get charges for a specific sample
sample_idx = 0
charges_sample = hirshfeld[sample_idx]
print(f"Sample {sample_idx} Hirshfeld charges: {charges_sample}")

# Check if charges sum to zero (neutrality)
for i in range(len(hirshfeld)):
    total_charge = hirshfeld[i].sum()
    assert abs(total_charge) < 1e-6, f"Sample {i} not neutral!"
```

## Integration with Training Data

To add charges to your training dataset:

```python
import numpy as np

# Load training data
train = np.load('training_data_fixed/energies_forces_dipoles_train.npz')

# Load merged charges
charges = np.load('co2_charges_mp2.npz', allow_pickle=True)

# Match samples using R1, R2, angle metadata
# (Requires matching the geometries between datasets)

# Add charges to training data
enhanced_train = dict(train)
enhanced_train['charges_mk'] = charges['charges_mk'][:8000]  # Match train split
enhanced_train['charges_hirshfeld'] = charges['charges_hirshfeld'][:8000]

# Save enhanced dataset
np.savez_compressed('enhanced_train.npz', **enhanced_train)
```

## Notes

1. **All charge arrays should sum to ~0** (charge neutrality)
2. **Different methods give different values** - choose based on your needs:
   - **Hirshfeld**: Good general-purpose, physically motivated
   - **MK/CHELPG**: ESP-fitted, good for electrostatics
   - **MBIS**: Iterative, includes higher multipoles
   - **CM5**: Optimized for charge-dependent properties

3. **Units**: All charges in atomic units (electron charge, e)

4. **Padding**: If merging files with different numbers of atoms, arrays will be object type

## Troubleshooting

### "No files found"
```bash
# Check pattern
ls /path/to/files/*_charges.npz

# Try different pattern
python merge_charge_files.py /path/ --pattern "*.npz"
```

### "Variable shapes"
This is normal if you're merging different molecules. Charges will be stored as object arrays.

### "Missing charge method"
Some files may not contain all 8 charge methods. The script will only include methods present in all files.

