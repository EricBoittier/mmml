# `mmml pyscf-evaluate-mp2`

Batch MP2 evaluation.


## Usage

```bash
mmml pyscf-evaluate-mp2 --help
```

## Options

```text
usage: mmml pyscf-evaluate-mp2 [-h] -i INPUT [-o OUTPUT] [--method {dft,mp2}]
                               [--basis BASIS] [--xc XC] [--spin SPIN]
                               [--charge CHARGE] [--no-energy] [--no-gradient]
                               [--no-dipole] [--esp] [--esp-cpu-fallback]
                               [--EF] [--efield EFIELD]
                               [--efield-sigma EFIELD_SIGMA]
                               [--add-random-noise ADD_RANDOM_NOISE]
                               [--seed SEED]
                               [--no-efield-include-nuclear-energy]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT
  -o, --output OUTPUT
  --method {dft,mp2}
  --basis BASIS
  --xc XC
  --spin SPIN
  --charge CHARGE
  --no-energy
  --no-gradient
  --no-dipole
  --esp
  --esp-cpu-fallback
  --EF
  --efield EFIELD
  --efield-sigma EFIELD_SIGMA
  --add-random-noise ADD_RANDOM_NOISE
  --seed SEED
  --no-efield-include-nuclear-energy
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
