"""Tests for intra-monomer close-contact detection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_GEOM_PATH = Path(__file__).resolve().parents[2] / "karml" / "utils" / "geometry_checks.py"
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


def test_neighbor_search_matches_brute_and_periodic_minimum():
    """Solvent-sized monomers must use the neighbor query, including MIC pairs."""
    grid = np.array(
        np.meshgrid(
            np.arange(2.0, 10.0, 2.0),
            np.arange(2.0, 10.0, 2.0),
            np.arange(2.0, 10.0, 2.0),
            indexing="ij",
        )
    ).reshape(3, -1).T
    block = np.vstack([grid, grid + np.array([1.0, 0.0, 0.0])])
    assert len(block) > _geom._INTRA_BRUTE_MAX_ATOMS
    side = 12.0
    block = block.copy()
    block[0] = [0.2, 6.0, 6.0]
    block[1] = [side - 0.25, 6.0, 6.0]
    cell = np.diag([side, side, side])
    excluded = {(2, 3), (4, 5)}
    found = _geom.min_counted_intramonomer_pair(block, excluded, cell, None)
    brute = _geom._min_counted_pair_brute(block, excluded, cell, None)
    assert found is not None and brute is not None
    assert found[0] == pytest.approx(brute[0])
    assert found[0] == pytest.approx(0.45, abs=1.0e-6)
    assert {found[1], found[2]} == {0, 1}


def test_chunked_pair_scan_matches_brute():
    rng = np.random.default_rng(2)
    block = rng.normal(size=(40, 3))
    excluded = {(0, 1), (1, 2)}
    found = _geom._min_counted_pair_chunked(block, excluded, None, 0.2)
    brute = _geom._min_counted_pair_brute(block, excluded, None, 0.2)
    assert found is not None and brute is not None
    assert found[0] == pytest.approx(brute[0])


def test_inward_nudge_does_not_oscillate_a_monomer_wider_than_the_cell():
    """A rigid slide cannot seat both faces of a monomer that spans the box."""
    side = 10.0
    pos = np.array(
        [
            [0.2, 5.0, 5.0],
            [12.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3], dtype=int)
    cell = np.diag([side, side, side])
    once = _geom.ensure_monomers_inside_cell(pos, offsets, cell, margin_A=0.05)
    twice = _geom.ensure_monomers_inside_cell(once, offsets, cell, margin_A=0.05)
    assert once == pytest.approx(pos)
    assert twice == pytest.approx(once)


def test_inward_nudge_still_seats_a_small_monomer():
    side = 10.0
    pos = np.array(
        [
            [-0.4, 5.0, 5.0],
            [0.6, 5.0, 5.0],
            [0.1, 5.5, 5.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 3], dtype=int)
    cell = np.diag([side, side, side])
    got = _geom.ensure_monomers_inside_cell(pos, offsets, cell, margin_A=0.05)
    assert float(got[:, 0].min()) == pytest.approx(0.05)
    assert float(got[1, 0] - got[0, 0]) == pytest.approx(1.0)


def test_box_filling_monomer_ignores_opposite_face_pair():
    """A pair ~L apart is an image contact when the monomer is wider than the box."""
    side = 10.0
    pos = np.zeros((4, 3), dtype=float)
    pos[0] = [0.0, 5.0, 5.0]
    pos[1] = [10.4, 5.0, 5.0]
    pos[2] = [5.0, 5.0, 5.0]
    pos[3] = [6.5, 5.0, 5.0]
    offsets = np.array([0, 4], dtype=int)
    dist, violation = find_worst_intramonomer_close_contact(
        pos,
        offsets,
        set(),
        cell=np.diag([side, side, side]),
        min_distance=0.88,
    )
    assert violation is not None
    assert dist == pytest.approx(1.5)
    assert {violation.atom_i, violation.atom_j} == {2, 3}


def test_small_monomer_still_flags_a_wrapped_contact():
    side = 10.0
    pos = np.zeros((3, 3), dtype=float)
    pos[0] = [0.1, 5.0, 5.0]
    pos[1] = [9.8, 5.0, 5.0]
    pos[2] = [0.1, 6.5, 5.0]
    offsets = np.array([0, 3], dtype=int)
    dist, violation = find_worst_intramonomer_close_contact(
        pos,
        offsets,
        set(),
        cell=np.diag([side, side, side]),
    )
    assert violation is not None
    assert dist == pytest.approx(0.3, abs=1.0e-6)
    assert {violation.atom_i, violation.atom_j} == {0, 1}


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
