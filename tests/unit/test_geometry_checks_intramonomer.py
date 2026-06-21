"""Tests for intra-monomer close-contact detection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_GEOM_PATH = Path(__file__).resolve().parents[2] / "mmml" / "utils" / "geometry_checks.py"
_spec = importlib.util.spec_from_file_location("_test_geometry_checks_intra", _GEOM_PATH)
assert _spec is not None and _spec.loader is not None
_geom = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _geom
_spec.loader.exec_module(_geom)

build_bond_exclusion_pairs = _geom.build_bond_exclusion_pairs
find_worst_intramonomer_close_contact = _geom.find_worst_intramonomer_close_contact
assert_no_intramonomer_close_contact = _geom.assert_no_intramonomer_close_contact


def test_build_bond_exclusion_pairs_includes_1_3():
    ib, jb = [1, 2], [2, 3]
    excluded = build_bond_exclusion_pairs(ib, jb, exclude_1_3=True)
    assert (0, 1) in excluded
    assert (1, 2) in excluded
    assert (0, 2) in excluded


def test_separate_intramonomer_contacts_relieves_geminal_clash():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3, 5], dtype=int)
    excluded = build_bond_exclusion_pairs([1, 2], [2, 3], exclude_1_3=True)
    new_pos = _geom.separate_intramonomer_contacts(
        pos,
        offsets,
        excluded,
        min_distance=0.5,
        margin=0.05,
    )
    dist, violation = find_worst_intramonomer_close_contact(
        new_pos,
        offsets,
        excluded,
        min_distance=0.5,
    )
    assert violation is None or dist >= 0.5


def test_find_worst_intramonomer_close_contact_skips_bonded_pairs():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3, 5], dtype=int)
    excluded = build_bond_exclusion_pairs([1, 2], [2, 3], exclude_1_3=False)
    dist, violation = find_worst_intramonomer_close_contact(
        pos, offsets, excluded
    )
    assert violation is not None
    assert dist == pytest.approx(0.25)
    assert violation.monomer == 0
    assert {violation.atom_i, violation.atom_j} == {0, 2}


def test_assert_no_intramonomer_close_contact_ok_for_normal_geometry():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.36, 1.03, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3, 5], dtype=int)
    excluded = build_bond_exclusion_pairs([1, 2, 1], [2, 3, 3], exclude_1_3=True)
    dmin = assert_no_intramonomer_close_contact(
        pos, offsets, excluded, min_distance=1.0, context="test"
    )
    assert dmin >= 1.0


def test_assert_no_intramonomer_close_contact_raises():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3], dtype=int)
    excluded = build_bond_exclusion_pairs([1, 2], [2, 3], exclude_1_3=False)
    with pytest.raises(RuntimeError, match="intra-monomer close contact"):
        assert_no_intramonomer_close_contact(
            pos, offsets, excluded, min_distance=1.0, context="test"
        )


def test_collapsed_1_3_geminal_hh_still_flagged():
    """Geminal H–H (PSF 1–3) must not hide sub-threshold clashes."""
    pos = np.array(
        [
            [0.0, 0.0, 0.0],   # C
            [1.00, 0.0, 0.0],  # H
            [1.40, 0.0, 0.0],  # H (0.40 Å from geminal partner)
        ],
        dtype=float,
    )
    offsets = np.array([0, 3], dtype=int)
    excluded = build_bond_exclusion_pairs([1, 1], [2, 3], exclude_1_3=True)
    assert (0, 1) in excluded and (0, 2) in excluded and (1, 2) in excluded
    dist, violation = find_worst_intramonomer_close_contact(
        pos, offsets, excluded, min_distance=0.5
    )
    assert violation is not None
    assert dist == pytest.approx(0.40)
    with pytest.raises(RuntimeError, match="intra-monomer close contact"):
        assert_no_intramonomer_close_contact(
            pos, offsets, excluded, min_distance=0.5, context="test"
        )
