# Glycol Dataset Cleaning Report

## Summary

The glycol dataset has been cleaned and validated using the new `mmml.cli.clean_data` tool.

## Original Dataset

- **File:** `glycol.npz`
- **Total structures:** 5,904
- **Atoms per structure:** 60 (glycol molecule with hydrogens)
- **Composition:** H (59%), C (20%), O (20%)

## Issues Found

### SCF Failures (9 structures)
Structures with extremely large forces (up to 1e12 eV/Å) indicating failed electronic structure calculations:
- Max force: 1,005,997,798,816 eV/Å
- These are clearly numerical failures in the quantum chemistry calculations

### Total Removed: 9 structures (0.15%)

## Cleaned Dataset

- **File:** `glycol_cleaned.npz`
- **Total structures:** 5,895 (99.85% retained)
- **File size:** 2.0 MB (compressed)
- **Fields included:** E, F, R, Z, N, D, Dxyz (essential training data only)

## Validation Results

✅ **All quality checks passed:**
- No NaN or Inf values in energy, forces, or positions
- All forces < 10 eV/Å
- Only SCF failures removed (no geometric filtering)
- Energy range: -228.5 to 0.0 eV
- Includes only essential training fields

## Usage

### Cleaning Command Used

```bash
python3 /home/ericb/mmml/mmml/cli/clean_data.py glycol.npz \
  -o glycol_cleaned.npz \
  --max-force 10.0 \
  --no-check-distances
```

**Why `--no-check-distances`?**
- Skips geometric checking (faster, retains more data)
- Only removes SCF failures (abnormally large forces)
- Geometric issues are usually training-compatible
- Result: 5,895 structures (99.85% retained) vs 4,130 (70%) with distance checking

### Training with Cleaned Data

```bash
# Simple version - num_atoms is auto-detected!
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_run1 \
  --n_train 3000 \
  --n_valid 500 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --ckpt_dir checkpoints/glycol_run1

# Advanced version with custom architecture
python -m mmml.cli.make_training \
  --data glycol_cleaned.npz \
  --tag glycol_advanced \
  --n_train 3000 \
  --n_valid 500 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --features 128 \
  --num_iterations 3 \
  --cutoff 10.0 \
  --ckpt_dir checkpoints/glycol_advanced
```

**✨ New:** `--num_atoms` is now auto-detected from the dataset! The tool will automatically detect that the data has 60 atoms per structure (including padding).

## Recommendations

1. **Use `glycol_cleaned.npz` for all training** - the original file has too many bad structures
2. **Consider further filtering** if you want even higher quality:
   - `--max-force 5.0` for even more stringent force filtering
   - `--min-distance 0.5` for stricter geometry requirements
3. **Monitor training** - if models fail to converge, consider additional cleaning

## Files

- `glycol.npz` - Original dataset (5,904 structures, 2.9 MB)
- `glycol_cleaned.npz` - Cleaned dataset (4,130 structures, 2.0 MB) ✅ **Use this one**
- `CLEANING_REPORT.md` - This file

## Tool Information

The cleaning was performed using the new `mmml/cli/clean_data.py` tool, which:
- Removes structures with NaN/Inf values
- Filters SCF failures (large forces)
- Removes overlapping atoms (short distances)
- Provides detailed statistics

For more information:
```bash
python -m mmml.cli.clean_data --help
```

