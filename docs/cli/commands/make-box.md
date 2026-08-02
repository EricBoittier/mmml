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
                     [--packmol-region {box,sphere}]
                     [--packmol-tolerance PACKMOL_TOLERANCE]
                     [--packmol-nloop PACKMOL_NLOOP]
                     [--fill-fraction FILL_FRACTION] [--no-packmol-pbc]

Pack a solute into a periodic box (vacuum copies or explicit solvent). Stages
--pdb to pdb/initial.pdb when given, then runs Packmol + CHARMM.

Input & configuration:
  --pdb PDB             Solute PDB (CGenFF residue/atom names). Copied to
                        pdb/initial.pdb before Packmol. When omitted,
                        pdb/initial.pdb must already exist (e.g. from mmml make-
                        res).

Scientific model:
  --density DENSITY     Solvent (or neat liquid) density in kg/m³. Built-in for
                        TIP3/water (1000) and OCOH/octanol (824); required for
                        other solvents when sizing N from density.

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --n N
  --res RES             Tag for output files (psf/system-<res>.psf,
                        pdb/init-<res>.pdb).
  --side_length, --box-size SIDE_LENGTH
                        Cubic box side length in Å. '--box-size' is an alias,
                        matching the naming used elsewhere in the CLI suite.
  --solvent SOLVENT     CGenFF solvent residue name (any RESI in
                        top_all36_cgenff.rtf), e.g. TIP3, MEOH, ACO, OCOH, ACN,
                        DMSO. Aliases: water→TIP3, octanol→OCOH.
  --packmol-region {box,sphere}
                        Solvent placement region. 'box' (default) fills the
                        cubic cell outside a solute exclusion sphere — this
                        matches the L³ volume used to size N from --density.
                        'sphere' packs a droplet shell instead, which holds only
                        pi/6 (52%) of the cell.
  --packmol-tolerance PACKMOL_TOLERANCE
                        Packmol minimum interatomic distance in Å (default 2.0).
                        Lower to relax packing.
  --packmol-nloop PACKMOL_NLOOP
                        Packmol GENCAN loops per molecule type (default 200;
                        Packmol's own default is 50).
  --fill-fraction FILL_FRACTION
                        Fraction of ideal bulk-density occupancy to request
                        (default 0.98). N is clamped to this; lower it if
                        Packmol still fails to converge.
  --no-packmol-pbc      Disable Packmol's 'pbc' keyword. By default the cell is
                        packed periodically so bulk density has no clashes with
                        periodic images.
```

## Visual examples

![Packed acetone box (Packmol)](../../images/structures/make-box-acetone.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)
- [Liquid box workflow](../../liquid-box-workflow.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
