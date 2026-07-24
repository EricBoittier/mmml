# `mmml npz2traj`

NPZ → ASE trajectory (E/F/dipole/charges).


## Usage

```bash
mmml npz2traj --help
```

## Options

```text
usage: mmml npz2traj [-h] -o OUTPUT [--max-structures MAX_STRUCTURES]
                     [--stride STRIDE] [--start START] [--ase-units] [--quiet]
                     input

Convert MMML NPZ datasets to ASE trajectories with energy, forces, dipole,
charges, and extra fields attached for GUI inspection.

positional arguments:
  input                 Input NPZ file

Input & configuration:
  --max-structures MAX_STRUCTURES
                        Maximum number of structures to convert

Scientific model:
  --ase-units           Convert E/F/D from NPZ schema units to ASE calculator
                        units (eV, eV/Å, e·Å). Without this flag, values stay in
                        NPZ units and unit labels are stored in atoms.info.

Output & artifacts:
  -o, --output OUTPUT   Output trajectory (.traj, .extxyz, .xyz, …)

Diagnostics & safety:
  -h, --help            show this help message and exit
  --quiet               Suppress progress output

Other options:
  --stride STRIDE       Use every Nth structure (default: 1)
  --start START         First structure index (default: 0)

Examples: mmml npz2traj data.npz -o trajectory.traj mmml npz2traj data.npz -o
subset.traj --max-structures 100 --stride 10 mmml npz2traj data.npz -o
frames.extxyz mmml npz2traj data.npz -o ase.traj --ase-units Schema keys: R/Z
required; E, F, D/Dxyz, N, mono, cell optional. Default units follow NPZ schema
(E Hartree, F Hartree/Bohr, D Debye). Use --ase-units to convert calculator +
info energy/forces/dipole to ASE conventions (eV, eV/Å, e·Å).
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
