"""Live CHARMM: reloading the CGenFF parameters must not disable van der Waals.

Regression guard for the sticky ``READ PARAM APPEND`` in
``setup/charmm/source/api/api_read.F90``. ``read_param_file`` declared its append
flag with an initializer::

    logical :: qappend = .false., qflex = .false.
    if (append .ne. 0) qappend = .true.

A Fortran local declared with an initializer is implicitly ``SAVE``d, and the
body only ever *set* the flag. So the first ``read.prm(..., append=True)``
latched append mode for the life of the process, and the next full parameter
read silently ran as ``READ PARAM APPEND``, wiping CHARMM's live NONBONDED
table. Every later energy then had ``VDWaals == 0``.

That is not a theoretical concern: ``read_cgenff_toppar()`` appends the bundled
``examples/m/par_ch3cl.prm`` whenever it is present, and the Packmol cluster
builder calls ``read_cgenff_toppar()`` twice — once per monomer template, once in
``_build_cluster_psf_from_composition``. Clusters were therefore minimized with
no repulsion at all: ABNR converged to an electrostatic collapse that stretched
TIP3 O-H from 0.953 A to 1.257 A.

Two properties are pinned here, cheaply (one dimer, one ENER each), because the
symptom is silent — CHARMM reports ``VDWaals  -0.00000`` and exits 0.

Note for anyone simplifying the fixture: the base parameter read **must** use
``flex=True`` (as ``read_cgenff_prm`` does). ``read.prm`` without it takes a
different flag combination through ``parmio`` and does not reproduce the
failure, so a "tidied" fixture that drops flex silently stops testing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import can_import_pycharmm

if not can_import_pycharmm():
    pytest.skip("PyCHARMM is not available", allow_module_level=True)

pytestmark = pytest.mark.pycharmm

# TIP3 dimer at 2.8 A: close enough that VDW is unambiguously non-zero, far
# enough that nothing is pathological.
_MONOMER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]])
_DIMER = np.vstack([_MONOMER, _MONOMER + np.array([2.8, 0.0, 0.0])])


def _vdw_probe():
    """Return a callable giving CHARMM's VDW energy for a fresh TIP3 dimer PSF."""
    import pandas as pd
    import pycharmm
    import pycharmm.coor as coor
    import pycharmm.energy as energy
    import pycharmm.generate as gen
    import pycharmm.psf as psf
    import pycharmm.read as read
    import pycharmm.settings as settings

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        pycharmm_quiet,
        reset_block,
    )

    pycharmm_quiet()
    reset_block()
    settings.set_bomb_level(-5)

    def vdw() -> float:
        psf.delete_atoms()
        for seg in ("A", "B"):
            read.sequence_string("TIP3")
            gen.new_segment(seg)
        coor.set_positions(pd.DataFrame(_DIMER, columns=["x", "y", "z"]))
        pycharmm.lingo.charmm_script("ENER")
        return float(energy.get_vdw())

    return vdw


def test_full_parameter_read_after_an_append_keeps_vdw_alive():
    """A full (append=False) parameter read must replace, never append.

    Minimal expression of the api_read.F90 defect: base read, one append read,
    base read again. On a libcharmm with the saved ``qappend`` the third read
    runs as an append and the VDW table is gone.
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_paths import (
        assert_cgenff_toppar_readable,
    )
    from mmml.interfaces.pycharmmInterface.cgenff_residues import extra_cgenff_prm_paths
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        _rtf_path_without_drude_autogen,
        read_cgenff_prm,
    )

    toppar = assert_cgenff_toppar_readable()
    extra = extra_cgenff_prm_paths()
    if not extra:
        pytest.skip("no bundled append PRM in this checkout; nothing to latch")

    vdw = _vdw_probe()

    read.rtf(_rtf_path_without_drude_autogen(toppar.rtf))
    read_cgenff_prm(prm_path=toppar.prm, bomlev=False)
    base = vdw()
    assert abs(base) > 1e-6, (
        f"degenerate fixture: VDW is {base} straight after the first parameter "
        "read, so this test cannot detect the table being wiped later"
    )

    for path in extra:
        read_cgenff_prm(prm_path=path, append=True, bomlev=False)
    assert vdw() == pytest.approx(base, rel=1e-9), (
        "appending the bundled residue PRM changed the base VDW energy"
    )

    # The read under test: append=False, so it must fully replace the table.
    read.rtf(_rtf_path_without_drude_autogen(toppar.rtf))
    read_cgenff_prm(prm_path=toppar.prm, bomlev=False)
    after = vdw()

    assert after == pytest.approx(base, rel=1e-9), (
        f"VDW is {after} after re-reading the full CGenFF parameters, expected "
        f"{base}. Exactly 0.0 means the append=False read was executed as "
        "READ PARAM APPEND and zeroed the NONBONDED table -- the saved qappend "
        "in api_read.F90 read_param_file."
    )


def test_read_cgenff_toppar_is_idempotent():
    """``read_cgenff_toppar()`` twice in one session must leave VDW unchanged.

    This is the call pattern the Packmol cluster builder actually uses, and it
    is what silently produced monomer-distorting cluster minimizations.
    """
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    vdw = _vdw_probe()

    read_cgenff_toppar()
    first = vdw()
    assert abs(first) > 1e-6, (
        f"VDW is {first} after read_cgenff_toppar(). CHARMM is a per-process "
        "singleton, so on a libcharmm with the saved qappend this is the "
        "previous test's parameter reads having already wiped the NONBONDED "
        "table, not a bad fixture. Run this file alone to see it in isolation."
    )

    read_cgenff_toppar()
    second = vdw()
    assert second == pytest.approx(first, rel=1e-9), (
        f"VDW is {second} after a second read_cgenff_toppar(), expected {first}. "
        "0.0 means the repeated toppar load wiped CHARMM's NONBONDED table; any "
        "CHARMM minimization after this point runs with no repulsion."
    )

    # Third time: the latch, if present, only needs one append to arm, so a
    # third pass must be just as stable as the second.
    read_cgenff_toppar()
    assert vdw() == pytest.approx(first, rel=1e-9)
