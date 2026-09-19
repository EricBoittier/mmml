# `mmml tune-mm-nonbonded`

Fit CGenFF LJ/charge scales of the ML/MM tail to teacher liquid frames.


## Usage

```bash
mmml tune-mm-nonbonded --help
```

## Options

```text
usage: mmml tune-mm-nonbonded [-h] {label,fit} ...

Fit per-type CGenFF LJ (eps/Rmin) scales and a charge scale so student-ML + MM
reproduces teacher interaction energies/forces of liquid frames.

positional arguments:
  {label,fit}
    label      teacher/student interaction labels of box frames
    fit        fit scales, bootstrap, report cohesion budget

options:
  -h, --help   show this help message and exit
```


## Related docs

- [Trainable hybrid MM LJ scales](../../hybrid-mm-lj-scales.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
