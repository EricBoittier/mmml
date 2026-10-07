# `karml make-res`

CGENFF residue → PDB/PSF/topology.


## Usage

```bash
karml make-res --help
```

## Options

```text
usage: karml make-res [-h] [--res RES] [--list-residues] [--no-pager]
                     [--skip-energy-show]

Generate a CGENFF residue (PDB, PSF, topology) via PyCHARMM.

Input & configuration:
  --list-residues     List valid CGENFF residue names and descriptions (opens
                      less on a TTY).

Scientific model:
  --skip-energy-show  Skip the final CHARMM energy.show() (avoids segfault on
                      some clusters/SLURM).

Diagnostics & safety:
  -h, --help          show this help message and exit

Other options:
  --res RES           CGENFF residue name (RESI in top_all36_cgenff.rtf), e.g.
                      ACO, CYBZ, TIP3.
  --no-pager          With --list-residues, print the table to stdout instead of
                      piping to less.

Examples: karml make-res --list-residues karml make-res --list-residues --no-pager
| grep -i acetone karml make-res --res ACO
```

## Visual examples

![Acetone monomer (ACO)](../../images/structures/make-res-aco.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
