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
  --density DENSITY     Solvent (or neat liquid) density in kg/m³. Built-in for
                        TIP3/water (1000) and OCOH/octanol (824); required for
                        other solvents when sizing N from density.

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --n N
  --res RES
  --side_length, --box-size SIDE_LENGTH
                        Cubic box side length in Å. '--box-size' is an alias,
                        matching the naming used elsewhere in the CLI suite.
  --solvent SOLVENT     CGenFF solvent residue name (any RESI in
                        top_all36_cgenff.rtf), e.g. TIP3, MEOH, ACO, OCOH.
                        Aliases: water→TIP3, octanol→OCOH.
```

## Example structures

![Packed acetone box (Packmol)](../../images/structures/make-box-acetone.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)
- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
