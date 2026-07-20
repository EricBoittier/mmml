# `mmml dimer-scan`

Reproducible rigid 1D dimer energy/force scan.


## Usage

```bash
mmml dimer-scan --help
```

## Options

```text
usage: mmml dimer-scan [-h] --calculator {physnet,xtb} [--checkpoint CHECKPOINT]
                       [--distance START:STOP:STEP]
                       [--energy-definition {interaction,total}]
                       [--charge CHARGE] [--spin SPIN] [--seed SEED]
                       [--allow-partial] [--overwrite] --output OUTPUT
                       RESIDUE [RESIDUE ...]

Run a reproducible rigid 1D dimer energy/force scan.

positional arguments:
  RESIDUE               One residue for a homodimer or two for a heterodimer

options:
  -h, --help            show this help message and exit
  --calculator {physnet,xtb}
                        Explicit ASE calculator type
  --checkpoint CHECKPOINT
                        PhysNet checkpoint path
  --distance START:STOP:STEP
                        Inclusive distance grid in angstrom (default:
                        2.5:6.0:0.1)
  --energy-definition {interaction,total}
  --charge CHARGE
  --spin SPIN
  --seed SEED
  --allow-partial
  --overwrite
  --output OUTPUT
```


## Related docs

- [1D dimer scan design](../../dimer-scan-design.md)
- [Scientific code policy](../../scientific-code.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
