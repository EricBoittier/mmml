"""Tests for pycharmm.cdocker.FACTS_rescore — repeatability of FACTS scoring.

Confirms that FACTS state is properly cleared between successive scoring
calls (a known historical bug). The test runs FACTS_rescore 10 times on
identical input coordinates and asserts that all 10 returned energies
are equal to within numerical tolerance.

Two variants:
  - All atoms free
  - A random subset of atoms fixed
"""

import numpy as np
import pytest

from pycharmm import cdocker, coor, gen, ic, psf, read, select_atoms, settings


@pytest.fixture(scope="module")
def alanine_capped():
    """Build a single-residue capped alanine peptide."""
    settings.set_bomb_level(-1)
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36m_prot.prm", flex=True)
    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()


def _facts_rescore_repeated(fix_sel, n_trials=10):
    """Run FACTS_rescore n_trials times from the same coords; return energies."""
    xyz_0 = coor.get_positions()
    energies = []
    for _ in range(n_trials):
        coor.set_positions(xyz_0)
        energies.append(cdocker.FACTS_rescore(fixAtomSel=fix_sel, steps=100, tolgrd=0.001))
    return np.asarray(energies)


def _assert_repeatable(energies, what):
    """Fail with the whole sequence, saying which call went wrong.

    Reported separately from the drift check because a non-finite energy is a
    different failure with a different cause, and "max delta = nan" says
    neither which call produced it nor what the others were -- which is
    exactly the position an intermittent CI failure left us in.
    """
    bad = np.flatnonzero(~np.isfinite(energies))
    assert bad.size == 0, (
        f"FACTS returned a non-finite energy {what}: "
        f"call(s) {bad.tolist()} of {energies.size} gave "
        f"{energies[bad].tolist()}; full sequence {energies.tolist()}"
    )
    assert np.allclose(energies, energies[0], atol=1e-5), (
        f"FACTS energies drift across repeated calls {what}: "
        f"max delta = {np.max(np.abs(energies - energies[0])):.2e}; "
        f"full sequence {energies.tolist()}"
    )


def test_facts_clear_all_free(alanine_capped):
    """FACTS state clears between calls when all atoms are free."""
    energies = _facts_rescore_repeated(select_atoms.SelectAtoms())
    _assert_repeatable(energies, "with all atoms free")


def test_facts_clear_with_fixed_atoms(alanine_capped):
    """FACTS state clears between calls when some atoms are fixed."""
    # Random subset to be FREE; the rest are fixed
    rng = np.random.default_rng(seed=0)
    select = rng.choice([True, False], size=psf.get_natom())
    fix_sel = select_atoms.SelectAtoms().set_selection(select)
    energies = _facts_rescore_repeated(fix_sel)
    _assert_repeatable(energies, "with some atoms fixed")
