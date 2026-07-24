# `mmml make-box`

Pack molecules into a periodic box.


## Usage

```bash
mmml make-box --help
```

## Options

```text
usage: mmml make-box [-h] [--n N] [--res RES] [--side_length SIDE_LENGTH]
                     [--pdb PDB] [--solvent SOLVENT] [--density DENSITY]

Input & configuration:
  --pdb PDB

Scientific model:
  --density DENSITY

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --n N
  --res RES
  --side_length, --box-size SIDE_LENGTH
                        Cubic box side length in Å. '--box-size' is an alias,
                        matching the naming used elsewhere in the CLI suite.
  --solvent SOLVENT
```

## Example structures

![Packed acetone box (Packmol)](../../images/structures/make-box-acetone.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)
- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
