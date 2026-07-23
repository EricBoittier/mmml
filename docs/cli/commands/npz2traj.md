# `mmml npz2traj`

Convert MMML NPZ datasets to ASE trajectories with energy, forces, dipole,
charges, and extra fields attached for GUI inspection.


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

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output trajectory (.traj, .extxyz, .xyz, …)
  --max-structures MAX_STRUCTURES
                        Maximum number of structures to convert
  --stride STRIDE       Use every Nth structure (default: 1)
  --start START         First structure index (default: 0)
  --ase-units           Convert E/F/D from NPZ schema units to ASE calculator
                        units (eV, eV/Å, e·Å). Without this flag, values stay
                        in NPZ units and unit labels are stored in atoms.info.
  --quiet               Suppress progress output
```

## What gets attached

| NPZ key | ASE destination | Notes |
|---------|-----------------|-------|
| `R`, `Z` | positions / numbers | Padding removed via `N` and `Z>0` |
| `E` | `atoms.info['energy']` + calculator | Default Hartree |
| `F` | `atoms.arrays['forces']` + calculator | Default Hartree/Bohr |
| `D` / `Dxyz` | `atoms.info['dipole']` + calculator | Default Debye |
| `mono` / `Q` | `atoms.arrays['charges']` | Also kept under original key |
| `cell` / `box` | `atoms.cell` + PBC | Optional |
| metadata | `atoms.info['npz_*']` | method, units, source path, … |

Use `--ase-units` when opening the trajectory in ASE’s GUI and you want force
arrows / energies in eV and eV/Å.

## Examples

```bash
mmml npz2traj data.npz -o trajectory.traj
mmml npz2traj data.npz -o subset.traj --max-structures 100 --stride 10
mmml npz2traj data.npz -o frames.extxyz
mmml npz2traj data.npz -o ase.traj --ase-units
```
