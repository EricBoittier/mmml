"""Tests for the pycharmm.energy introspection API.

Build an alanine dipeptide, minimize, then exercise the energy module's
property/term getters:
  - get_property_names / get_property_statuses / get_properties
  - get_term_names / get_term_statuses / get_terms
  - get_energy (DataFrame)
  - get_property_by_name

Original by C. L. Brooks III, April 2019.
"""

import pytest

from pycharmm import (
    NonBondedScript,
    coor,
    energy,
    gen,
    ic,
    minimize,
    read,
    settings,
)


@pytest.fixture(scope="module")
def minimized_alanine_dipeptide():
    """Build alanine dipeptide and run a short minimization.

    Starts by calling :func:`pycharmm.reset.everything` so the fixture
    is robust against state leaked by tests that ran before this
    module in the pytest collection order.  Specifically: a
    ``NBOND ... IMGFRQ N`` set by an earlier dynamics test will
    otherwise survive into our ``minimize.run_abnr`` call, where the
    default ``INBFRQ=50`` may not be a divisor of N, tripping
    ``FINCYC: IMGFRQ is not a multiple of INBFRQ`` and killing the
    Python process before this test can finish.  ``reset.everything``
    drops the prior atoms, restores default cutoffs / frequencies, and
    is a no-op on a clean session.
    """
    from pycharmm import reset

    reset.everything()

    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)

    read.rtf("data/water_ions.rtf", append=True)
    read.prm("data/water_ions.prm", append=True, flex=True)

    old_warn = settings.set_warn_level(-1)
    old_bomb = settings.set_bomb_level(-1)
    read.prm("data/sodium_oxygen_nbfixes.prm", append=True, flex=True)
    settings.set_warn_level(old_warn)
    settings.set_bomb_level(old_bomb)

    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()

    coor.orient(by_rms=False, by_mass=False, by_noro=False)

    NonBondedScript(
        cutnb=18.0,
        ctonnb=15.0,
        ctofnb=13.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    ).run()

    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)


def test_get_property_names(minimized_alanine_dipeptide):
    names = energy.get_property_names()
    assert "ener" in [n.lower() for n in names]
    assert "grms" in [n.lower() for n in names]


def test_get_term_names(minimized_alanine_dipeptide):
    names = energy.get_term_names()
    # CHARMM always reports BOND/ANGL/DIHE/VDW/ELEC for a built system.
    upper = [n.upper() for n in names]
    for needed in ("BOND", "ANGL", "DIHE"):
        assert needed in upper, f"Missing energy term {needed} in {upper}"


def test_get_energy_dataframe(minimized_alanine_dipeptide):
    etable = energy.get_energy()
    # Should contain ENER + GRMS columns.
    assert "ENER" in etable.columns
    assert "GRMS" in etable.columns
    assert len(etable) >= 1


def test_get_property_by_name_matches_dataframe(minimized_alanine_dipeptide):
    etable = energy.get_energy()
    ener_via_api = energy.get_property_by_name("ener")
    grms_via_api = energy.get_property_by_name("grms")

    # Float comparison via approximate equality
    assert abs(ener_via_api - float(etable.ENER.iloc[0])) < 1e-6
    assert abs(grms_via_api - float(etable.GRMS.iloc[0])) < 1e-6
