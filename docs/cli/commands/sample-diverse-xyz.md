# `mmml sample-diverse-xyz`

Pick diverse structures (SOAP) → NPZ.


## Usage

```bash
mmml sample-diverse-xyz --help
```

## Options

```text
usage: mmml sample-diverse-xyz [-h] -n N [-o OUTPUT] [--seed SEED]
                               [--species SPECIES] [--r-cut R_CUT]
                               [--n-max N_MAX] [--l-max L_MAX] [--sigma SIGMA]
                               [-v]
                               inputs [inputs ...]

Save top-N diverse structures from XYZ files to sampled.npz (SOAP space).

positional arguments:
  inputs                One or more multi-frame XYZ files (same stoichiometry
                        and atom order).

Input & configuration:
  -n, --n-structures N  Number of diverse structures to keep.

Execution:
  --seed SEED           RNG seed for the first farthest-point seed (default: 0).

Output & artifacts:
  -o, --output OUTPUT   Output NPZ path (default: sampled.npz).

Diagnostics & safety:
  -h, --help            show this help message and exit
  -v, --verbose

Other options:
  --species SPECIES     Comma-separated chemical symbols for SOAP (default:
                        H,C,O).
  --r-cut R_CUT
  --n-max N_MAX
  --l-max L_MAX
  --sigma SIGMA
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
