# Superseded: the CHARMM box-building route

These scripts built the solvated box through CHARMM and Packmol
(`mmml make-box`). They are kept because they document real, non-obvious fixes,
but **nothing on the live path uses them** — `jaxmd_box.py` replaced the whole
route, and the runtime no longer needs CHARMM at all.

| file | what it did | why it was replaced |
|---|---|---|
| `04_make_solvent_boxes.sh` | solvated the solute in each solvent with `mmml make-box` | `make-box` sizes the molecule count from the cube volume but packs into the inscribed sphere, giving a 2.2x over-dense box (CHARMM reported 7.5e22 kcal/mol) |
| `05_export_solute.py` | wrote a strictly column-aligned CGenFF solute PDB | only needed to feed `make-box`; `jaxmd_box.py` builds coordinates directly |
| `06_solvated_md.py` | one solvated trajectory, with a CHARMM composition build and solvent-cavity carving | 544 lines of which four helpers were live; those moved to `../solute.py`, and the CHARMM build path is gone |

`06_solvated_md.py` also recorded why moving solvent aside does not work: at
liquid density there is nowhere to move to. Random relocation left 431 of 461
waters clashing, and radial pushing ended with two atoms exactly coincident and
E = -9e5 eV. Deleting overlapping solvent is the workable approach, and
`jaxmd_box.py` avoids the problem entirely by placing the solute first.

Two further findings are still worth knowing, and are recorded in the main
README:

- The PDB residue-name field is columns 18–21, i.e. **four characters**. The
  five-character `CH3CL` forced a shifted layout that strict readers reject, so
  the residue was renamed `MECL` (see `../top_mecl.rtf`).
- CHARMM builds a PSF by reading the sequence from the PDB, which requires each
  residue's atoms to be **contiguous**. The canonical ML atom order interleaves
  the two solute residues, and Packmol responds by renumbering the split
  residue to 9999.

`../top_mecl.rtf` and `../top_chex.rtf` are **not** legacy: `10_extract_solvent_params.py`
still uses CHARMM offline to extract solvent parameters, and cyclohexane is only
available through `top_chex.rtf`.
