# `mmml unwrap-traj`

Unwrap periodic trajectories.


## Usage

```bash
mmml unwrap-traj --help
```

## Options

```text
usage: mmml unwrap-traj [-h] -o OUTPUT [--format {auto,traj,xyz,extxyz,dcd}]
                        [--fast] [--index INDEX] [--cell CELL]
                        [--group-size GROUP_SIZE] [--n-groups N_GROUPS]
                        [--no-molecules] [--reference REFERENCE]
                        [--coord-key COORD_KEY] [--numbers-key NUMBERS_KEY]
                        [--cell-key CELL_KEY] [--quiet]
                        input

Unwrap periodic ASE trajectories or HDF5 coordinate files.

positional arguments:
  input                 Input trajectory (.traj/.xyz/.extxyz/etc.) or .h5/.hdf5
                        file

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output file
  --format {auto,traj,xyz,extxyz,dcd}
                        Output format (default: infer from suffix)
  --fast                Use direct streaming writer for xyz/extxyz outputs
  --index INDEX         ASE input frame index/slice for non-HDF5 inputs
                        (default: :)
  --cell CELL           Override cell as 'a,b,c' or 9 matrix values
  --group-size GROUP_SIZE
                        Atoms per contiguous molecule/group for molecule-wise
                        unwrapping
  --n-groups N_GROUPS   Number of equal contiguous molecule/groups for molecule-
                        wise unwrapping
  --no-molecules        Disable automatic bonded-fragment grouping; unwrap atoms
                        independently
  --reference REFERENCE
                        ASE-readable file supplying atomic numbers and fallback
                        cell for HDF5 inputs
  --coord-key COORD_KEY
                        HDF5 coordinate dataset key (default:
                        R/positions/coordinates/coords/xyz)
  --numbers-key NUMBERS_KEY
                        HDF5 atomic-number dataset key (default:
                        Z/atomic_numbers/numbers)
  --cell-key CELL_KEY   HDF5 cell dataset key (default:
                        cell/cells/lattice/lattices/box/boxes)
  --quiet               Suppress summary output

Unwrap periodic trajectories and write ASE/XYZ outputs. Examples: mmml unwrap-
traj in.traj -o unwrapped.traj mmml unwrap-traj in.traj -o unwrapped.xyz
--format xyz --fast mmml unwrap-traj coords.h5 -o unwrapped.extxyz --reference
wrapped.traj --fast
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
