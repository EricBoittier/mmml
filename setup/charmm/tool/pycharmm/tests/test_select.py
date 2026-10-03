"""Tests for SelectAtoms — boolean selection by atom type / segment / etc.

Builds a small multi-segment system and exercises the SelectAtoms API:
  - constructor with atom_type kwarg
  - .by_atom_type chaining (single + union)
  - intersection with by-segment selection
  - .by_res_and_type with multi-value strings (segment / resid / atom)
  - select.bonded() to expand a selection by one bond shell

Original by C. L. Brooks III, November 2020.
"""

import numpy as np
import pytest

from pycharmm import SelectAtoms, gen, read, select


@pytest.fixture(scope="module")
def multi_segment_system():
    """Build ALAD (ALA GLY), GLAD (GLU), SAD (SER) capped peptides.

    Module-scoped because CHARMM topology state is global; rebuilding
    inside each test would accumulate duplicate segments.
    """
    read.rtf("data/top_all22_prot.inp")
    read.prm("data/par_all22_prot.inp")
    for seq, seg in [("ALA GLY", "ALAD"), ("GLU", "GLAD"), ("SER", "SAD")]:
        read.sequence_string(seq)
        gen.new_segment(seg_name=seg, first_patch="ACE", last_patch="CT3", setup_ic=True)


def _count(sel):
    return np.sum(list(sel))


def test_select_atom_type_kwarg(multi_segment_system):
    """SelectAtoms(atom_type='CA') yields the four CA atoms."""
    assert _count(SelectAtoms(atom_type="CA")) == 4


def test_select_by_atom_type_chain(multi_segment_system):
    """.by_atom_type('CA') matches the constructor form."""
    assert _count(SelectAtoms().by_atom_type("CA")) == 4


def test_select_atom_type_intersect_segment(multi_segment_system):
    """CA atoms restricted to segment ALAD (2 residues, 2 CAs)."""
    sel = SelectAtoms().by_atom_type("CA") & SelectAtoms(seg_id="ALAD")
    assert _count(sel) == 2


def test_select_by_atom_type_union(multi_segment_system):
    """.by_atom_type chained twice gives the union (CA atoms + C atoms)."""
    assert _count(SelectAtoms().by_atom_type("CA").by_atom_type("C")) == 8


def test_by_res_and_type_single(multi_segment_system):
    """by_res_and_type with single segment/resid/atom selects 1 atom."""
    assert _count(SelectAtoms().by_res_and_type("ALAD", "1", "CA")) == 1


def test_by_res_and_type_multi_atom(multi_segment_system):
    """Multiple atom types in one call → union of those atoms."""
    assert _count(SelectAtoms().by_res_and_type("ALAD", "1", "CY N CA C")) == 4


def test_by_res_and_type_multi_segid(multi_segment_system):
    """Multiple segids in one call → union across the named segments."""
    assert _count(SelectAtoms().by_res_and_type("ALAD GLAD", "1", "CY N CA C")) == 8


def test_by_res_and_type_multi_resid(multi_segment_system):
    """Multiple resids in one call → union across the named residues."""
    assert _count(SelectAtoms().by_res_and_type("ALAD", "1 2", "CY N CA C")) == 7


def test_select_bonded_expands_by_one_shell(multi_segment_system):
    """select.bonded(atoms) returns atoms + their bonded neighbors."""
    base = SelectAtoms().by_res_and_type("ALAD", "1", "CA")
    expanded = SelectAtoms(select.bonded(base))
    assert _count(expanded) == 5


def test_select_and_deselect_single_atom(multi_segment_system):
    atoms = SelectAtoms(update=False)
    assert not atoms._select_atom(2)
    assert atoms[2] and _count(atoms) == 1
    assert atoms._deselect_atom(2)
    assert not np.any(atoms)
