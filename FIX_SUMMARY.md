# Energy Preprocessing Fix Summary

## Problem

The `--subtract-atomic-energies` flag was not working correctly. Training was showing:
1. Extremely high losses (~10^18)
2. Incorrect energy values in validation data (e.g., [-94, 148] eV instead of expected ~0 eV)
3. Training divergence

## Root Cause

**Data Leakage:** Atomic energies were being computed separately for training and validation datasets, rather than computing them once on training data and applying the same references to validation data.

This violated the fundamental ML principle: preprocessing parameters must be fit on training data only, then applied to validation/test data.

## Solution

Fixed the trainer to:
1. Compute atomic energies from training data only
2. Extract these atomic energies from training metadata
3. Apply the SAME atomic energies to validation data
4. Prevent recomputation on validation set

## Changes Made

### 1. Fixed `trainer.py` (lines 251-303)

Added logic to:
- Compute atomic energies on training data
- Extract them from metadata
- Apply them to validation data without recomputation
- Display computed atomic energies for transparency

### 2. Added Automatic Weight Adjustment (lines 320-346)

When energy scale changes significantly due to preprocessing, automatically adjusts loss weights to maintain training stability. (Note: For CO2 dataset with uniform composition, this isn't triggered since std doesn't change).

### 3. Added Verbose Output

Shows:
- Which atomic energies were computed
- Energy and force statistics after preprocessing
- Whether automatic weight adjustment was applied

## Verification

### Test Results

```bash
python test_energy_preprocessing.py
```

All tests pass ✅:
- Unit conversion
- Atomic energy computation
- Per-atom scaling  
- Full pipeline with real CO2 data

### Training with Atomic Subtraction

```bash
python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --subtract-atomic-energies \
  --no-tensorboard
```

**Results:**
- ✅ Atomic energies correctly computed: C = -1020.88 eV, O = -2041.76 eV
- ✅ Energies correctly transformed: [-3.49, 5.45] eV (binding energies)
- ✅ Training proceeds normally with reasonable losses
- ✅ Forces unchanged (correct physics)

## Current Status

### Working Features

✅ Energy unit conversion (eV, Hartree, kcal/mol, kJ/mol)  
✅ Atomic energy reference subtraction (linear regression or mean)  
✅ Per-atom energy scaling  
✅ Energy normalization  
✅ Proper train/validation split for preprocessing  
✅ Metadata tracking of all transformations  

### Usage Examples

**Basic atomic energy subtraction:**
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --subtract-atomic-energies
```

**With per-atom scaling:**
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --subtract-atomic-energies \
  --scale-by-atoms
```

**Convert from Hartree:**
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --energy-unit hartree \
  --convert-energy-to eV \
  --subtract-atomic-energies
```

## Important Notes

### For Uniform Composition Datasets (like CO2-only)

When all molecules have the same composition:
- Energy std will **not change** after atomic subtraction (this is correct!)
- All molecules get the same constant subtracted
- Variance comes entirely from binding energy (what we want)
- Main benefit is interpretability (binding vs absolute energies)

### For Mixed Composition Datasets

When molecules have different compositions:
- Energy scale may change significantly
- Automatic weight adjustment may trigger
- Model learns to predict binding energies across compositions
- Better generalization expected

## Documentation

New documentation files:
- `ENERGY_PREPROCESSING.md`: Complete guide with examples
- `ATOMIC_ENERGY_SUBTRACTION_README.md`: Detailed explanation
- `QUICK_REFERENCE.md`: Quick command reference
- `test_energy_preprocessing.py`: Comprehensive test suite

Example scripts:
- `train_with_atomic_refs.sh`: Training with atomic energy subtraction
- `train_per_atom.sh`: Training with per-atom scaling

## Known Issues

1. **TensorBoard Import Error:** Unrelated to our changes, use `--no-tensorboard` flag as workaround
2. **First NPZ Load Bug:** Fixed! Changed `allow_pickle=True` to `False` in `loaders.py`

## Backward Compatibility

✅ All changes are backward compatible:
- Default behavior unchanged (no preprocessing)
- Existing scripts work without modification
- New options are opt-in via command-line flags

## Testing

Run comprehensive tests:
```bash
cd examples/co2/physnet_train
python test_energy_preprocessing.py
```

Expected output:
```
✅ All tests passed!
  unit_conversion: ✅ PASS
  atomic_energies: ✅ PASS
  scale_by_atoms: ✅ PASS
  full_pipeline: ✅ PASS
```

## Files Modified

### Core Implementation
- `mmml/data/preprocessing.py`: +176 lines (new functions)
- `mmml/data/loaders.py`: +67 lines (DataConfig extension)
- `examples/co2/physnet_train/trainer.py`: +85 lines (CLI args and logic)

### Testing & Documentation
- `examples/co2/physnet_train/test_energy_preprocessing.py`: New (278 lines)
- `examples/co2/physnet_train/ENERGY_PREPROCESSING.md`: New (233 lines)
- `examples/co2/physnet_train/ATOMIC_ENERGY_SUBTRACTION_README.md`: New
- `examples/co2/physnet_train/QUICK_REFERENCE.md`: New (71 lines)
- `examples/co2/physnet_train/train_with_atomic_refs.sh`: New
- `examples/co2/physnet_train/train_per_atom.sh`: New

## Next Steps

The energy preprocessing functionality is now fully working and tested. You can:

1. **Use atomic energy subtraction** for learning binding energies:
   ```bash
   python trainer.py --train train.npz --valid valid.npz --subtract-atomic-energies
   ```

2. **Test with your own data** - the feature works with any molecular dataset

3. **Combine preprocessing options** as needed for your specific use case

4. **Check metadata** after loading to see all applied transformations:
   ```python
   data = load_npz('file.npz', config=config)
   print(data['metadata'][0])
   ```

## Summary

✅ **Fixed:** Atomic energy subtraction now works correctly  
✅ **Added:** Proper train/validation separation for preprocessing  
✅ **Added:** Automatic weight adjustment for scale changes  
✅ **Added:** Comprehensive documentation and tests  
✅ **Verified:** All tests pass, training works correctly  

