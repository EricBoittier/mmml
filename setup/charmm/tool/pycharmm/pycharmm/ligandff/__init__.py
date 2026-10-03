"""Build CHARMM force-field files for small-molecule ligands from AMBER GAFF2.

Turns a SMILES string or an sdf file into a CHARMM residue-topology (rtf) and
parameter (prm) file, plus a CHARMM-readable pdb, by running the GAFF2 /
AM1-BCC route (antechamber, parmchk2, tleap) and converting the resulting AMBER
topology. Based on C. L. Brooks III's conversion workflow.

Typical use -- build the files, then load them alongside a protein force field:

>>> from pycharmm import ligandff
>>> lig = ligandff.build_ligand_ff("CC(=O)Oc1ccccc1C(O)=O", resname="AIN")
>>> ligandff.load_ligand_ff(lig, segid="AIN")            # doctest: +SKIP

``build_ligand_ff`` writes files and returns their paths; it does not touch
CHARMM state. ``load_ligand_ff`` is the opt-in step that reads them in.

GAFF atom types are written with a leading ``_`` so a ligand force field can be
loaded next to the protein force field (e.g. parm19sb in
``toppar/non_charmm``) without atom-type collisions.

Requirements
------------
External programs, needed only when building: ``antechamber``, ``parmchk2``,
``tleap`` (AmberTools). Python packages: ``parmed``, and ``rdkit`` for SMILES
input. These are optional dependencies of pycharmm -- importing this module
never requires them, and a missing one is reported only when the code path that
needs it runs.

Notes
-----
The parm19sb protein-side builder is deliberately not part of this public API
yet: it converts a single prmtop and has no driver for assembling a whole
protein force field from many of them. It is still reachable for that work as
``from pycharmm.ligandff.protein_ff import build_protein_forcefield``.
"""

from .charmm_load import load_ligand_ff, load_ligand_ffs
from .pipeline import (
    LigandFF,
    LigandSet,
    build_ligand_ff,
    combine_ligand_ffs,
)

__all__ = [
    "LigandFF",
    "LigandSet",
    "build_ligand_ff",
    "combine_ligand_ffs",
    "load_ligand_ff",
    "load_ligand_ffs",
]
