"""Query the loaded residue topology file (RTF).

Truman asked (via Charlie) for pyCHARMM access to the RTF's residue
names/count and atom-type names/count, so a script can check whether the
topology it loaded actually contains the residues and atom types it
needs.  These tests cover the ``pycharmm.rtf`` wrappers over the
``api_rtf`` C-API.
"""

import pytest

from pycharmm import read, reset, rtf, settings


@pytest.fixture
def all22_rtf():
    """A clean CHARMM with only top_all22_prot loaded."""
    reset.everything()
    old_bomb = settings.set_bomb_level(-2)
    old_warn = settings.set_warn_level(-2)
    try:
        read.rtf("data/top_all22_prot.inp")
    finally:
        settings.set_bomb_level(old_bomb)
        settings.set_warn_level(old_warn)
    return None


def test_residue_names_present(all22_rtf):
    """The protein RTF's residues are listed, and count matches the list."""
    names = rtf.get_residue_names()
    assert rtf.get_num_residues() == len(names)
    assert len(names) > 0
    # A few residues every protein topology must define.
    for expected in ("ALA", "GLY", "TRP"):
        assert expected in names, f"{expected} missing from {names}"


def test_atom_type_names_present(all22_rtf):
    """The protein RTF's atom types are listed, and count matches the list."""
    types = rtf.get_atom_type_names()
    assert rtf.get_num_atom_types() == len(types)
    assert len(types) > 0
    for expected in ("NH1", "CT1", "C", "O"):
        assert expected in types, f"{expected} missing from {types}"


def test_names_are_clean(all22_rtf):
    """Names come back stripped of the Fortran field padding, none empty."""
    for n in rtf.get_residue_names() + rtf.get_atom_type_names():
        assert n == n.strip() and n != ""


def test_rtf_info_is_consistent(all22_rtf):
    """get_rtf_info() bundles exactly the residue and atom-type lists."""
    info = rtf.get_rtf_info()
    assert set(info) == {"residues", "atom_types"}
    assert info["residues"] == rtf.get_residue_names()
    assert info["atom_types"] == rtf.get_atom_type_names()


def test_query_is_consistent_after_reset():
    """After a reset the queries are safe to call and stay self-consistent
    (count always equals the length of the returned list, even when the
    RTF arrays are unallocated)."""
    reset.everything()
    assert rtf.get_num_residues() == len(rtf.get_residue_names())
    assert rtf.get_num_atom_types() == len(rtf.get_atom_type_names())
