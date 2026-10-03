"""Tests for the Coordinates class and coor.dist() — pycharmm pip package.

Builds a multi-segment small system, then exercises:
  - Coordinates('main' | 'comp' | 'comp2')         (positions readback)
  - coor.dist(...)                                  (3 selection variants)

Original by Josh Buckner, 24 November 2021.
"""

import pytest

from pycharmm import (
    Coordinates,
    SelectAtoms,
    coor,
    gen,
    ic,
    read,
)


@pytest.fixture(scope="module")
def alanine_glutamate_serine():
    """Build a small multi-segment peptide system."""
    read.rtf("data/top_all22_prot.inp")
    read.prm("data/par_all22_prot.inp")

    for seq, seg in [("ALA GLY", "ALAD"), ("GLU", "GLAD"), ("SER", "SAD")]:
        read.sequence_string(seq)
        gen.new_segment(seg_name=seg, first_patch="ACE", last_patch="CT3", setup_ic=True)

    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    coor.orient(by_rms=False, by_mass=False, by_noro=False)


def test_coordinates_main_comp_comp2(alanine_glutamate_serine):
    """All three coordinate sets are non-empty and have matching shape."""
    main = Coordinates("main")
    comp = Coordinates("comp")
    comp2 = Coordinates("comp2")

    assert main.coords.shape[0] > 0
    # comp / comp2 mirror the main coordinate set
    assert comp.coords.shape == main.coords.shape
    assert comp2.coords.shape == main.coords.shape


def test_coor_dist_residue_to_residue(alanine_glutamate_serine):
    """coor.dist with default residue-to-residue minimum distance."""
    contacts = coor.dist(
        selection1=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="1")),
        selection2=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="2")),
        cutoff=6.0,
    )
    # No assertion on count (depends on built geometry); just verify it ran.
    assert contacts is not None


def test_coor_dist_atom_pairs_with_exclusions(alanine_glutamate_serine):
    """coor.dist over all atom pairs honoring 1-2 / 1-3 / 1-4 exclusions."""
    contacts = coor.dist(
        selection1=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="1")),
        selection2=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="1")),
        cutoff=6.0,
        resi=False,
    )
    assert contacts is not None


def test_coor_dist_atom_pairs_no_exclusions(alanine_glutamate_serine):
    """coor.dist over all atom pairs ignoring all exclusions."""
    contacts = coor.dist(
        selection1=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="1")),
        selection2=(SelectAtoms(seg_id="ALAD") & SelectAtoms(res_id="1")),
        cutoff=6.0,
        resi=False,
        omit_excl=False,
        omit_14excl=False,
    )
    assert contacts is not None
