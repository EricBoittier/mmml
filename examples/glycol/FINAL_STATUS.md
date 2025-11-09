# Glycol Dataset - Final Status

## ✅ ALL ISSUES RESOLVED

### Final Cleaning Results

**Original dataset:**
- 5,904 structures
- Issues: 9 SCF failures + 113 zero energies

**Cleaned dataset:**
- **5,782 structures** (97.9% retained)
- **122 problematic structures removed** (2.1%)
  - 113 with E=0 (failed calculations)
  - 9 with extreme forces (SCF failures)

**Energy distribution (cleaned):**
- Range: [-228.52, -226.76] eV ✅
- Mean: -227.90 eV ✅
- No zeros, no positive values ✅

### Fields Included

✅ **Essential training fields:**
- E (energies)
- F (forces)
- R (positions)
- Z (atomic numbers)
- N (number of atoms)
- D (dipoles)
- Dxyz (dipole components)

❌ **Removed QM fields:**
- orbital_occupancies, orbital_energies
- cube_density_*, cube_potential_*
- metadata

### Train/Valid/Test Splits

- **Train:** 4,625 samples (80%)
- **Valid:** 578 samples (10%)
- **Test:** 579 samples (10%)
- **Seed:** 42 (reproducible)

All splits have only essential fields ✅

### Final Workflow

```bash
# 1. Clean (removes 122 bad structures)
python3 /home/ericb/mmml/mmml/cli/clean_data.py glycol.npz \
    -o glycol_cleaned.npz \
    --max-force 10.0 \
    --max-energy -1.0 \
    --no-check-distances

# 2. Split (80/10/10)
python3 /home/ericb/mmml/mmml/cli/split_dataset.py glycol_cleaned.npz -o splits/

# 3. Train (auto-detects num_atoms=60)
python3 /home/ericb/mmml/mmml/cli/make_training.py \
    --data splits/data_train.npz \
    --tag glycol_production \
    --n_train 4000 \
    --n_valid 500 \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --features 128 \
    --num_iterations 3 \
    --cutoff 10.0 \
    --ckpt_dir checkpoints/glycol_production
```

## Key Improvements to clean_data.py

### New Feature: Energy Validation

**Added checks for:**
- ✅ Zero energies (E ≈ 0)
- ✅ Suspiciously high energies (E > -1 eV by default)
- ✅ Suspiciously low energies (E < -1e6 eV by default)

**Usage:**
```bash
# Default (catches E ≈ 0 and E > -0.001)
python -m mmml.cli.clean_data data.npz -o clean.npz

# Custom energy thresholds
python -m mmml.cli.clean_data data.npz -o clean.npz \\
    --max-energy -10.0 --min-energy -10000
```

## Summary

| Metric | Value |
|--------|-------|
| **Original structures** | 5,904 |
| **Removed** | 122 (2.1%) |
| **Final structures** | 5,782 |
| **Zero energies** | 0 ✅ |
| **SCF failures** | 0 ✅ |
| **Energy range** | [-228.52, -226.76] eV ✅ |
| **Fields** | 7 essential fields ✅ |
| **Train/Valid/Test** | 4625/578/579 ✅ |

## Production Ready! 🚀

The glycol dataset is now fully cleaned, validated, and ready for training:

- ✅ No failed calculations
- ✅ All energies in valid range
- ✅ Dipole fields included
- ✅ Only essential fields
- ✅ Properly split
- ✅ Auto-detection works

**Start training now!**

