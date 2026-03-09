# Atomic Energy Subtraction - Important Notes

## Summary

The atomic energy subtraction feature is **now working correctly**. This document explains how it works, when to use it, and important considerations.

## What It Does

Atomic energy subtraction transforms molecular energies from absolute values to binding/interaction energies:

```
E_binding = E_molecular - Σ(E_atomic_i)
```

Where `E_atomic_i` are reference energies for each element, computed automatically from your training data using linear regression.

## Example: CO2 Dataset

### Before Atomic Energy Subtraction:
```
Energy mean: -5104.41 eV
Energy std:  1.77 eV
Energy range: [-5107.89, -5098.45] eV
```

### After Atomic Energy Subtraction:
```
Energy mean: -0.01 eV  (centered on binding energy)
Energy std:  1.77 eV   (SAME - see note below)
Energy range: [-3.49, 5.45] eV

Computed atomic energies:
  C (Z=6): -1020.8817 eV
  O (Z=8): -2041.7634 eV
```

## ⚠️ Important: Standard Deviation for Uniform Composition

**For datasets where all molecules have the same composition (like CO2-only datasets), the standard deviation will NOT change after atomic energy subtraction.**

This is **correct behavior** because:
1. All molecules have 1 C + 2 O atoms
2. Subtracting the same constant (C_energy + 2×O_energy) from all energies
3. Standard deviation is invariant to adding/subtracting constants
4. The variance comes entirely from binding energy differences, which is what we want!

## How It Works Correctly

1. **Training Set:** Atomic energies are computed from training data only using linear regression
2. **Validation Set:** The SAME atomic energies from training are applied (not recomputed)
3. **This prevents data leakage** and ensures proper validation

## Usage

```bash
python trainer.py \
  --train train.npz \
  --valid valid.npz \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression
```

## When To Use

✅ **Use when:**
- You want to learn binding/interaction energies instead of absolute energies
- Your dataset has molecules with varying compositions
- You want to remove the dominant atomic contribution to focus on chemistry
- You're comparing energies across different molecular sizes

❌ **Less useful when:**
- All molecules have exactly the same composition (e.g., only CO2)
  - Still works, but main benefit is interpretability (binding energies vs absolute)
  - Won't change the relative difficulty of learning

## Benefits

1. **Better Interpretability:** Binding energies are more chemically meaningful
2. **Improved Transferability:** Atomic references help models generalize
3. **Focuses Learning:** Model learns chemical interactions, not atomic energies
4. **Standard Practice:** Used in many ML potential papers (SchNet, PhysNet, etc.)

## Technical Details

### Linear Regression Method

The atomic energies are computed by solving:
```
E_molecular = n_C × E_C + n_O × E_O + n_H × E_H + ...
```

For multiple molecules with different compositions using least-squares regression.

For the CO2 dataset, this gives:
- C: -1020.88 eV
- O: -2041.76 eV

Which can be compared to DFT atomic energies for validation.

### Forces Remain Unchanged

Forces are energy gradients and are **not affected** by atomic energy subtraction:
```
F = -∇E_total = -∇(E_binding + E_atomic) = -∇E_binding
```

Since atomic energies are constants, their gradient is zero. This is correct physics!

## Example Output

```
Energy preprocessing:
  Input unit: eV
  Subtracting atomic energies: linear_regression

Loading training data...
  Structures: 8000
  
  Atomic energies computed from training data:
    C (Z=6): -1020.8817 eV
    O (Z=8): -2041.7634 eV

Loading validation data...
  Structures: 1000

✅ Data loaded:
  Energy std: 1.7739 eV
  Force std: 8.3630 eV/Å

Extra Validation Info:
E: Array[1000] x∈[-3.489, 5.452] μ=-0.012 σ=1.777 cpu:0
```

## Testing

Run the test suite to verify:
```bash
python test_energy_preprocessing.py
```

All tests should pass, including the full pipeline test with real CO2 data.

## References

- [PhysNet Paper](https://doi.org/10.1021/acs.jctc.8b00908) - Uses atomic energy references
- [SchNet Paper](https://arxiv.org/abs/1706.08566) - Atomic energy decomposition
- [GDML Paper](https://www.science.org/doi/10.1126/sciadv.1603015) - Force fields and atomic references

