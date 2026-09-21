# `mmml fit-liquid`

Fit ML/MM LJ scales and dimer λ to experimental ρ(T) / ΔHvap(T).


## Usage

```bash
mmml fit-liquid --help
```

## Options

```text
usage: mmml fit-liquid [-h] {dry-run,decompose,check,fit} ...

Fit per-type CGenFF LJ scales (and a PhysNet dimer scale) to experimental rho(T)
/ dHvap(T) by trajectory reweighting.

positional arguments:
  {dry-run,decompose,check,fit}
    dry-run             toy decompose + check + a few Adam steps on NPT frames
    decompose           evaluate hybrid terms per frame and write a cache
    check               same-Hamiltonian check of a cache at theta_0
    fit                 Adam on the reweighted liquid-observable loss

options:
  -h, --help            show this help message and exit
```


## Related docs

- [Trainable hybrid MM LJ scales](../../hybrid-mm-lj-scales.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
