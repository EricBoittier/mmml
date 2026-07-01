# `mmml fix-and-split`

Unit fixes + train/valid/test splits.


## Usage

```bash
mmml fix-and-split --help
```

## Options

```text
usage: mmml fix-and-split [-h] [--coords-in {auto,bohr,angstrom}] [--coords-out {angstrom,bohr,same}] [--energy-in {hartree,ev}] [--energy-out {ev,hartree,same}] [--force-in {hartree-bohr,ev-angstrom}] [--force-out {ev-angstrom,hartree-bohr,same}]
                          [--dipole-in {debye,e-angstrom}] [--dipole-out {e-angstrom,debye,same}] [--grid-coords-in {auto,bohr,angstrom,index}] [--grid-coords-out {angstrom,bohr,same}] [--preserve-units] --efd FILE [FILE ...] [--grid GRID] --output-dir OUTPUT_DIR
                          [--train-frac TRAIN_FRAC] [--valid-frac VALID_FRAC] [--test-frac TEST_FRAC] [--seed SEED] [--cube-spacing CUBE_SPACING] [--skip-validation] [--atomic-ref SCHEME] [--atomic-ref-units {hartree,ev}] [--n-grid-points N] [--esp-sd-sigma N]
                          [--esp-max-abs-kcal-mol X] [--min-dist-to-atoms Å] [--quiet] [--flip-forces] [--energy-scale X] [--zscale-energies] [--force-scale X] [--dipole-scale X] [--efield-scale X] [--esp-scale X] [--charge-scale X]

Fix units and create train/valid/test splits from molecular NPZ data

options:
  -h, --help            show this help message and exit
  --efd, --energies-forces-dipoles FILE [FILE ...]
                        Path(s) to energies_forces_dipoles.npz file(s). Multiple files are concatenated.
  --grid, --grids-esp GRID
                        Path to grids_esp.npz file (optional)
  --output-dir, -o OUTPUT_DIR
                        Directory to save output files
  --train-frac TRAIN_FRAC
                        Fraction of data for training (default: 0.8)
  --valid-frac VALID_FRAC
                        Fraction of data for validation (default: 0.1)
  --test-frac TEST_FRAC
                        Fraction of data for testing (default: 0.1)
  --seed SEED           Random seed for reproducible splits (default: 42)
  --cube-spacing CUBE_SPACING
                        Grid spacing in Bohr from original cube files (default: 0.25)
  --skip-validation     Skip validation checks
  --atomic-ref SCHEME   Subtract per-atom reference energies (e.g. pbe0/sz, pbe0/def2-tzvp)
  --atomic-ref-units {hartree,ev}
                        Units of refs in JSON: hartree (pbe0/def2-tzvp) or ev (pbe0/sz may use eV; default: hartree)
  --n-grid-points N     Target number of ESP grid points per sample (default 3000). Excludes tails (±SD) and points near atoms.
  --esp-sd-sigma N      Exclude grid points beyond ±N SD from mean (default 3.0, ignores distribution tails)
  --esp-max-abs-kcal-mol X
                        Exclude grid points with |esp| > X kcal/mol/e (default 100.0)
  --min-dist-to-atoms Å
                        Exclude grid points closer than this to any atom in Å (default 1.0)
  --quiet, -q           Suppress detailed output
  --flip-forces         Negate F before unit conversion (Ha/Bohr → eV/Å). Use when F stores the PySCF energy gradient ∂E/∂R instead of forces F = −∇E. mmml pyscf-evaluate NPZ already uses −gradient.
  --energy-scale X      Multiply E by X after Hartree→eV. Also efield_energy, efield_scf_energy if present (default 1.0).
  --zscale-energies, --z-scale-energies
                        Z-scale E after creating splits using training-set mean/std: E = (E_eV - mean_train) / std_train. Saves energy_zscale_stats.json.
  --force-scale X       Multiply F by X after Ha/Bohr→eV/Å. Also efield_scf_F if present (default 1.0).
  --dipole-scale X      Multiply Dxyz by X after Debye→e·Å if present. Also scales D, efield_scf_D, efield_D (Debye) if present.
  --efield-scale X      Multiply Ef, efield_Ef, efield_scf_Ef by X if present (default 1.0).
  --esp-scale X         Multiply esp by X [Hartree/e] on grid splits and on EFD if esp is stored there (default 1.0).
  --charge-scale X      Multiply NPZ key Q by X if present. For total molecular charge when Q stores charge. PySCF ESP export may use Q for quadrupole—verify your file (default 1.0).

Unit conversion:
  Declare units in the NPZ files. Defaults assume PySCF/atomic units on input and ASE-style training units on output. Use --preserve-units or *-out same to avoid converting fields that are already correct.

  --coords-in {auto,bohr,angstrom}
                        Units of R in the input NPZ (default: auto = infer from bond lengths)
  --coords-out {angstrom,bohr,same}
                        Units of R in output NPZ (default: angstrom; same = no conversion)
  --energy-in {hartree,ev}
                        Units of E in the input NPZ (default: hartree)
  --energy-out {ev,hartree,same}
                        Units of E in output NPZ (default: ev; same = no conversion)
  --force-in {hartree-bohr,ev-angstrom}
                        Units of F in the input NPZ (default: hartree-bohr)
  --force-out {ev-angstrom,hartree-bohr,same}
                        Units of F in output NPZ (default: ev-angstrom; same = no conversion)
  --dipole-in {debye,e-angstrom}
                        Units of Dxyz in the input NPZ (default: debye)
  --dipole-out {e-angstrom,debye,same}
                        Units of Dxyz in output NPZ (default: e-angstrom; same = no conversion)
  --grid-coords-in {auto,bohr,angstrom,index}
                        Units of vdw_surface/vdw_grid/esp_grid (default: auto)
  --grid-coords-out {angstrom,bohr,same}
                        Grid coordinate units in output NPZ (default: angstrom; same = no conversion)
  --preserve-units      Force all *-out flags to same (no R/E/F/D/grid conversions). Overrides explicit --dipole-out, --energy-out, etc. For selective conversion, omit this flag and set *-out same per field.

Examples:
  # Basic usage with default 8:1:1 split (with grid)
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./training_data
  
  # Without grid data (EFD only)
  mmml fix-and-split --efd data.npz --output-dir ./training_data
  
  # Custom split ratios
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./training_data \
    --train-frac 0.7 --valid-frac 0.15 --test-frac 0.15
  
  # Different cube spacing (e.g., 0.5 Bohr)
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./training_data \
    --cube-spacing 0.5
  
  # Skip validation for speed
  mmml fix-and-split --efd data.npz --output-dir ./training_data \
    --skip-validation

  # Concatenate multiple NPZ files (e.g. extend training set with MD samples)
  mmml fix-and-split --efd train.npz md_evaluated.npz --output-dir ./splits_extended

  # NPZ has raw PySCF gradient in F (not forces): negate before converting to eV/Å
  mmml fix-and-split --efd raw.npz --output-dir ./out --flip-forces

  # Correct a systematic factor after normal conversion (e.g. duplicate unit fix upstream)
  mmml fix-and-split --efd data.npz --output-dir ./out --energy-scale 0.5 --force-scale 1.0

  # Z-scale energies with training-set statistics and save the mean/std
  mmml fix-and-split --efd data.npz --output-dir ./out --zscale-energies

  # Already in training units (eV, eV/Å, e·Å, Å): split only
  mmml fix-and-split --efd data.npz -o ./splits --preserve-units

  # Explicit: PySCF Hartree/Bohr in, ASE units out (same as default)
  mmml fix-and-split --efd pyscf.npz -o ./out \
    --energy-in hartree --energy-out ev \
    --force-in hartree-bohr --force-out ev-angstrom \
    --dipole-in debye --dipole-out e-angstrom
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
