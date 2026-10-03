"""Tests for FACTS_rescore on a benzene + T4 lysozyme docking system.

Verifies that FACTS_rescore reproduces known reference energies for:
  - the bound receptor (fixed receptor + benzene in pocket)
  - the binding energy (delta vs. ligand translated 500 Å away)
"""

import numpy as np
import pytest

from pycharmm import (
    SelectAtoms,
    cdocker,
    coor,
    gen,
    lingo,
    read,
    settings,
)

EXPECTED_RECEPTOR_ENERGY = -4419.27931
EXPECTED_BINDING_ENERGY = -15.60511
TOLERANCE = 0.01


@pytest.fixture
def benzene_t4_complex():
    """Build the benzene-T4 lysozyme complex (no minimization).

    Function-scope: each FACTS_rescore call below minimizes the ligand
    in place, so the receptor and binding tests must each start from a
    freshly-built complex.
    """
    # Reset any state from a prior test in the module so the fresh
    # build is not appended to a leftover PSF.
    lingo.charmm_script("dele atom sele all end")

    settings.set_bomb_level(-1)
    read.rtf("data/top_all36_prot.rtf")
    read.rtf("data/top_all36_cgenff.rtf", append=True)
    read.prm("data/par_all36m_prot.prm", flex=True)
    read.prm("data/par_all36_cgenff.prm", append=True, flex=True)
    settings.set_bomb_level(0)
    lingo.charmm_script("stream data/benzene.rtf")

    read.psf_card("data/t4.psf", append=True)
    read.pdb("data/t4.pdb", resid=True)
    read.sequence_pdb("data/benzene.pdb")
    gen.new_segment(seg_name="LIGA")
    read.pdb("data/benzene.pdb", resid=True)


@pytest.mark.stateful
def test_facts_receptor_energy(benzene_t4_complex):
    """Bound-state FACTS energy matches the historical reference value."""
    receptor = SelectAtoms().by_seg_id("LIGA").__invert__()
    energy = cdocker.FACTS_rescore(fixAtomSel=receptor, steps=100)
    assert abs(energy - EXPECTED_RECEPTOR_ENERGY) <= TOLERANCE, (
        f"FACTS receptor energy = {energy:.5f}, "
        f"expected {EXPECTED_RECEPTOR_ENERGY:.5f} "
        f"(tol = {TOLERANCE})"
    )


@pytest.mark.stateful
def test_facts_binding_energy(benzene_t4_complex):
    """Binding energy (bound − far-apart) matches the historical reference."""
    ligand = SelectAtoms().by_seg_id("LIGA")
    receptor = ligand.__invert__()

    bound = cdocker.FACTS_rescore(fixAtomSel=receptor, steps=100)

    # Translate the ligand 500 Å away (mask via the boolean SelectAtoms)
    pos = coor.get_positions()
    pos.x += 500 * np.array(ligand)
    coor.set_positions(pos)

    far = cdocker.FACTS_rescore(fixAtomSel=receptor, steps=0)
    binding = bound - far

    assert abs(binding - EXPECTED_BINDING_ENERGY) <= TOLERANCE, (
        f"FACTS binding energy = {binding:.5f}, "
        f"expected {EXPECTED_BINDING_ENERGY:.5f} "
        f"(tol = {TOLERANCE})"
    )
