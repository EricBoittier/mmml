"""Compare CHARMM ABNR and OpenMM minimization energies on alanine dipeptide.

After building the same alanine dipeptide system, minimize twice from
identical starting coordinates:
  - via CHARMM's ABNR minimizer
  - via OpenMM (minimize.run_omm)

The two final energies should agree within ~3e-4 (kcal/mol).

Original by C. L. Brooks III, April 2019.
"""

import numpy as np
import pytest

from pycharmm import blade, coor, energy, minimize, settings
from pycharmm.lingo import charmm_script

ENERGY_AGREEMENT_TOL = 3e-4


def test_run_blade_selects_supported_minimizers(monkeypatch):
    commands = []

    class CommandScript:
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))

        def run(self):
            return self

    monkeypatch.setattr("pycharmm.script.CommandScript", CommandScript)
    monkeypatch.setattr(blade, "check_interrupt", lambda: False)

    for method in ("lbfg", "sd", "sdfd", "sdmd"):
        assert minimize.run_blade(
            nstep=5, method=method, warn_restraints=False, handle_interrupt=False
        )

    assert [command for command, _ in commands] == [
        "mini blade lbfg",
        "mini blade sd",
        "mini blade sdfd",
        "mini blade sdmd",
    ]
    with pytest.raises(ValueError, match="Unknown BLaDE minimizer"):
        minimize.run_blade(method="abnr", warn_restraints=False, handle_interrupt=False)


# Leaves CHARMM/OpenMM context state that wedges later tests in the
# sweep. Run with `pytest -m stateful`.
@pytest.mark.stateful
def test_omm_and_abnr_minimization_agree(alanine_dipeptide_with_nbonds):
    """Final energy from run_omm and run_abnr agree within tolerance."""
    xyz_initial = coor.get_positions()

    # OpenMM minimization
    charmm_script("energy omm")
    minimize.run_omm(nstep=500, tolgrd=0)
    charmm_script("energy omm")
    eomm = energy.get_total()

    # ABNR minimization from the same starting structure
    coor.set_positions(xyz_initial)
    settings.set_verbosity(0)
    minimize.run_abnr(nstep=700, tolgrd=1e-3)
    settings.set_verbosity(5)
    eabnr = energy.get_total()

    assert np.abs(eomm - eabnr) <= ENERGY_AGREEMENT_TOL, (
        f"OpenMM and ABNR energies disagree: "
        f"E_omm = {eomm:.6f}, E_abnr = {eabnr:.6f}, "
        f"|delta| = {abs(eomm - eabnr):.6f} > tol = {ENERGY_AGREEMENT_TOL}"
    )
