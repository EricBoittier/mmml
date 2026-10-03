"""Test OpenMM and ABNR minimizers agree on alanine dipeptide.

Same intent as test_minimize.py but at a slightly different tolerance
(1.5e-3 vs. 3e-4) and includes a verbose path. Intended as the
"stricter" check that exercises full OpenMM force evaluation.

Skips when OpenMM isn't importable.

Original by C. L. Brooks III, April 2019.
"""

import numpy as np
import pytest

# OpenMM is required for this test (its primary subject).
openmm = pytest.importorskip("openmm")

from pycharmm import coor, energy, minimize, settings  # noqa: E402
from pycharmm.lingo import charmm_script  # noqa: E402

ENERGY_AGREEMENT_TOL = 1.5e-3


# Wedges later tests in the default sweep when an OpenMM context
# from a previous test is still alive. Run with `pytest -m stateful`.
@pytest.mark.stateful
def test_omm_and_abnr_minimizers_agree(alanine_dipeptide_with_nbonds):
    """OpenMM and ABNR minimization energies agree within 1.5e-3."""
    settings.set_verbosity(5)
    xyz = coor.get_positions()

    charmm_script("energy omm")
    minimize.run_omm(nstep=500, tolgrd=0)
    charmm_script("energy omm")
    eomm = energy.get_total()

    coor.set_positions(xyz)
    minimize.run_abnr(nstep=700, tolgrd=1e-3)
    eabnr = energy.get_total()

    assert np.abs(eomm - eabnr) <= ENERGY_AGREEMENT_TOL, (
        f"OpenMM and ABNR energies disagree: "
        f"E_omm = {eomm:.6f}, E_abnr = {eabnr:.6f}, "
        f"|delta| = {abs(eomm - eabnr):.6f} > tol = {ENERGY_AGREEMENT_TOL}"
    )
