# `mmml verify-esp-alignment`

Verify ESP grid alignment in NPZ.


## Usage

```bash
mmml verify-esp-alignment --help
```

## Options

```text
usage: mmml verify-esp-alignment [-h] -i INPUT [--sample SAMPLE]
                                 [--n-points N_POINTS] [--basis BASIS] [--xc XC]
                                 [--grid-in-angstrom]

Verify esp-grid alignment in evaluated NPZ (data generation check).

Input & configuration:
  -i, --input INPUT    Input NPZ (e.g. 07_evaluated.npz)

Scientific model:
  --basis BASIS        Basis (default def2-SVP)
  --xc XC              XC functional (default PBE0)

Diagnostics & safety:
  -h, --help           show this help message and exit

Other options:
  --sample SAMPLE      Sample index to check (default 0)
  --n-points N_POINTS  Number of grid points to verify (default 200, for speed)
  --grid-in-angstrom   Grid coords already in Angstrom (e.g. from fix-and-
                       split). Default: Bohr (pyscf-evaluate)

Verify esp-grid alignment at the data generation level. Recomputes ESP at grid
points using PySCF and compares to stored values. If aligned: high correlation.
If misaligned (bug in pyscf-evaluate): low correlation. Usage: mmml verify-esp-
alignment -i 07_evaluated.npz mmml verify-esp-alignment -i 07_evaluated.npz
--sample 0 --n-points 200
```

## Visual examples

![ESP from explicit distributed charge sites](../../images/povray-overlays/distributed_charge_esp.png)

![Distributed sites and equivalent multipoles](../../images/povray-overlays/distributed_charge_model.png)

## Related docs

- [DCMNet calculators and ESP](../../dcmnet_calculators.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
