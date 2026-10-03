"""Shared scaffolding for the test_custom_forces_* split test files.

The original single-file test_custom_forces.py grew to 26 classes and
1247 lines. It was split by force-family (basic / collective / misc)
for navigability; this helper holds the shared single-atom-system
setup so each split file can keep its own autouse fixture short.

The single-atom system is the smallest CHARMM scaffolding that lets
the OpenMM custom-force constructors run -- one fictitious atom of
mass 10, charge 0, no bonds. It's enough to instantiate Custom*Force
objects, set parameters on them, and verify the Python -> Fortran ->
C++ bridge round-trips correctly. Tests that need actual energy
evaluation should add forces to it and call energy.show().

Module-private (leading underscore) so pytest doesn't try to collect
it as a test file.
"""

from __future__ import annotations

import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, lingo, read
from pycharmm import generate as gen

_SINGLE_ATOM_RTF = """
    read rtf card
* Single atom topology file
*
   20    1
MASS     -1 X     10.0

RESI TEST       0.0
GROUP
ATOM A    X     0.0
PATC  FIRS NONE LAST NONE
END
"""

_SINGLE_ATOM_PRM = """
    read param card
* dummy parameters for testing
*
NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
X        0.0440    1.0       0.8000

END
"""


def setup_single_atom_system() -> None:
    """Build a minimal one-atom CHARMM system suitable for force tests.

    Idempotent at the level of "results in a one-atom MOL segment at
    the origin" -- if a previous fixture left atoms in the PSF, they
    are deleted first. Use as the body of a module-scope autouse
    fixture in each test_custom_forces_*.py file.
    """
    omm.clear()
    if psf.get_natom() > 0:
        lingo.charmm_script("delete atom sele all end")

    lingo.charmm_script(_SINGLE_ATOM_RTF)
    lingo.charmm_script(_SINGLE_ATOM_PRM)
    read.sequence_string("TEST")
    gen.new_segment(seg_name="MOL")
    pos = coor.get_positions()
    pos.iloc[0] = [0.0, 0.0, 0.0]
    coor.set_positions(pos)
