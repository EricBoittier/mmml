# `mmml interpolate-xyz`

Interpolate XYZ via Z-matrix → NPZ.


## Usage

```bash
mmml interpolate-xyz --help
```

## Options

```text
usage: mmml interpolate-xyz [-h] [-o OUTPUT] [--steps N] xyz1 xyz2

Interpolate between two XYZ files via Z-matrix coordinates and save frames to
NPZ (R, Z, N).

positional arguments:
  xyz1                 First XYZ file (defines Z-matrix connectivity)
  xyz2                 Second XYZ file (same atoms and ordering as the first)

options:
  -h, --help           show this help message and exit
  -o, --output OUTPUT  Output NPZ path (default: interpolated.npz)
  --steps N            Number of interpolation segments (N+1 frames written;
                       default: 1000)

CLI: interpolate between two XYZ geometries in internal (Z-matrix) coordinates.
Uses the first structure's Z-matrix topology; the second XYZ must match atom
order and count. Writes a compressed NPZ with R, Z, N per frame (same layout as
interpolate_xyzs_to_npz). Usage: mmml interpolate-xyz start.xyz end.xyz -o
path.npz --steps 500
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
