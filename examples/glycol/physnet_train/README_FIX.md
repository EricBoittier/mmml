# How to Fix Training

## Problem
You ran training **without energy preprocessing**, using raw absolute quantum energies (~-5104 eV).
This causes losses in the **billions** because the model can't handle such large values.

## Solution
Use **ANY** of these preprocessing options:

### Option 1: Subtract Atomic Energies (RECOMMENDED)
```bash
./train_fixed.sh
# or
./train_with_atomic_refs.sh
```

**Effect:** Converts absolute energies (-5104 eV) to interaction energies (±3 eV)
- Before: -5104.41 ± 1.77 eV
- After: 0.00 ± 1.77 eV

### Option 2: Scale by Number of Atoms
```bash
./train_per_atom.sh
```

**Effect:** Converts to per-atom energies
- Before: -5104.41 eV (total)
- After: ~-170 eV (per atom for 30-atom molecule)

### Option 3: Normalize
```bash
python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --normalize-energy \
  --epochs 100
```

**Effect:** Z-score normalization
- Mean → 0, Std → 1

## Quick Start (Run This Now)
```bash
./train_fixed.sh
```

This will train with proper energy preprocessing and you should see:
- Train Loss decreasing (not staying constant)
- Reasonable MAE values (not billions)
- Model actually learning!

## What Was Wrong
Your training output showed:
```
Energy preprocessing:
  Input unit: eV
  No preprocessing applied  ← PROBLEM!
```

You need to see:
```
Energy preprocessing:
  Input unit: eV
  Subtracting atomic energies: linear_regression  ← GOOD!
```

