# `mmml npz2traj`

NPZ → ASE trajectory (E/F/dipole/charges).


## Usage

```bash
mmml npz2traj --help
```

## Options

```text
usage: mmml npz2traj [-h] -o OUTPUT [--max-structures MAX_STRUCTURES]
                     [--stride STRIDE] [--start START] [--ase-units] [--psf PSF]
                     [--resnames RESNAMES] [--split-resnames SPLIT_RESNAMES]
                     [--dt-ps DT_PS] [--steps-per-frame STEPS_PER_FRAME]
                     [--quiet]
                     input

Convert MMML NPZ datasets to ASE trajectories with energy, forces, dipole,
charges, and extra fields attached for GUI inspection.

positional arguments:
  input                 Input NPZ file

Input & configuration:
  --max-structures MAX_STRUCTURES
                        Maximum number of structures to convert
  --psf PSF             CHARMM PSF matching NPZ atom order (required for .dcd
                        and for --resnames / --split-resnames). Copied or
                        subset-written next to each DCD.

Scientific model:
  --ase-units           Convert E/F/D from NPZ schema units to ASE calculator
                        units (eV, eV/Å, e·Å). Without this flag, values stay in
                        NPZ units and unit labels are stored in atoms.info.

Execution:
  --dt-ps DT_PS         DCD header timestep in ps (default: 1.0 if unset)
  --steps-per-frame STEPS_PER_FRAME
                        DCD NSAVC / steps between saved frames (default: 1)

Output & artifacts:
  -o, --output OUTPUT   Output trajectory (.traj, .extxyz, .xyz, .dcd, …)

Diagnostics & safety:
  -h, --help            show this help message and exit
  --quiet               Suppress progress output

Other options:
  --stride STRIDE       Use every Nth structure (default: 1)
  --start START         First structure index (default: 0)
  --resnames RESNAMES   Comma-separated residue names kept in the primary output
                        (e.g. TRIA or TIP3). Requires --psf.
  --split-resnames SPLIT_RESNAMES
                        Also write one trajectory (+PSF for .dcd) per residue
                        name, as {stem}.{RESNAME}{suffix}. Requires --psf.

Examples: mmml npz2traj data.npz -o trajectory.traj mmml npz2traj data.npz -o
subset.traj --max-structures 100 --stride 10 mmml npz2traj data.npz -o
frames.extxyz mmml npz2traj data.npz -o ase.traj --ase-units mmml npz2traj
nvt/trajectory.npz -o nvt/all.dcd --psf model.psf mmml npz2traj
nvt/trajectory.npz -o nvt/all.dcd --psf model.psf --split-resnames TRIA,TIP3
Schema keys: R/Z or positions/Z required; E, F, D, cell/boxes optional. Training
NPZs: default E Hartree / F Hartree/Bohr / D Debye (--ase-units → eV). jaxmd-
unified trajectory.npz: energies are eV; use --psf for .dcd.
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
