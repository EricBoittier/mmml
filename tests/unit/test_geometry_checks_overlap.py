"""Tests for inter-monomer overlap detection and separation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_GEOM_PATH = Path(__file__).resolve().parents[2] / "mmml" / "utils" / "geometry_checks.py"
_spec = importlib.util.spec_from_file_location("_test_geometry_checks", _GEOM_PATH)
assert _spec is not None and _spec.loader is not None
_geom = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _geom
_spec.loader.exec_module(_geom)

assert_no_intermonomer_atom_overlap = _geom.assert_no_intermonomer_atom_overlap
find_worst_intermonomer_overlap = _geom.find_worst_intermonomer_overlap
separate_intermonomer_overlaps = _geom.separate_intermonomer_overlaps
repack_monomers_clear_overlap = _geom.repack_monomers_clear_overlap
repack_selected_monomers_clear_overlap = _geom.repack_selected_monomers_clear_overlap
coords_pathological_for_repack = _geom.coords_pathological_for_repack


def test_find_worst_intermonomer_overlap_reports_closest_pair():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    dist, violation = find_worst_intermonomer_overlap(pos, offsets)
    assert violation is not None
    assert dist == 0.4
    assert violation.monomer_i == 0
    assert violation.monomer_j == 1
    assert violation.atom_i == 0
    assert violation.atom_j == 2


def test_separate_intermonomer_overlaps_restores_min_distance():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    out = separate_intermonomer_overlaps(
        pos, offsets, min_distance=1.5, margin=0.0, symmetric=True
    )
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=1.5, context="test"
    )
    assert dmin >= 1.5


def test_separate_intermonomer_overlaps_pbc_mic():
    pos = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [9.6, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    cell = np.diag([10.0, 10.0, 10.0])
    out = separate_intermonomer_overlaps(
        pos, offsets, min_distance=1.5, margin=0.0, cell=cell, symmetric=True
    )
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=1.5, cell=cell, context="pbc"
    )
    assert dmin >= 1.5


def test_repack_monomers_clear_overlap_entangled_pair():
    """Repack separates monomers that rigid COM push cannot fix easily."""
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    out = repack_monomers_clear_overlap(
        pos,
        offsets,
        min_distance=1.5,
        spacing=4.0,
        margin=0.0,
        seed=7,
    )
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=1.5, context="repack"
    )
    assert dmin >= 1.5


def test_repack_selected_monomers_preserves_fixed_monomer_coords():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4, 6], dtype=int)
    fixed_before = pos[2:4].copy()
    out = repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [0, 2],
        min_distance=1.5,
        margin=0.0,
        seed=1,
    )
    assert np.allclose(out[2:4], fixed_before)
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=1.5, context="selective_repack"
    )
    assert dmin >= 1.5


def test_push_apart_move_only_translates_selected_monomer():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    _, violation = find_worst_intermonomer_overlap(pos, offsets)
    assert violation is not None
    fixed = pos[2:4].copy()
    out = _geom.push_apart_overlapped_monomers(
        pos,
        offsets,
        violation,
        min_distance=1.5,
        margin=0.0,
        move_only=frozenset({1}),
    )
    assert np.allclose(out[0:2], pos[0:2])
    assert not np.allclose(out[2:4], fixed)


def test_repack_monomers_clear_overlap_pbc_dense():
    pos = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [9.6, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    cell = np.diag([10.0, 10.0, 10.0])
    out = repack_monomers_clear_overlap(
        pos,
        offsets,
        min_distance=1.5,
        spacing=3.0,
        margin=0.0,
        seed=3,
        cell=cell,
    )
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=1.5, cell=cell, context="repack_pbc"
    )
    assert dmin >= 1.5


def test_repack_monomers_caps_corrupted_internal_radius_for_pbc_pad():
    """Fly-off coords must not inflate PBC repack pad to astronomical values."""
    good = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    corrupted = good.copy()
    corrupted[2] = np.array([3.2e8, 0.0, 0.0])
    offsets = np.array([0, 2, 4], dtype=int)
    cell = np.diag([40.0, 40.0, 40.0])
    out = repack_monomers_clear_overlap(
        corrupted,
        offsets,
        min_distance=2.0,
        spacing=4.0,
        margin=0.0,
        seed=1,
        cell=cell,
        template_positions=good,
        max_monomer_radius_A=6.0,
    )
    assert np.all(np.isfinite(out))
    dmin = assert_no_intermonomer_atom_overlap(
        out, offsets, min_distance=2.0, cell=cell, context="corrupted_repack"
    )
    assert dmin >= 2.0


def test_coords_pathological_for_repack_detects_nonfinite_and_huge_radius():
    offsets = np.array([0, 2, 4], dtype=int)
    ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    assert not coords_pathological_for_repack(ok, offsets, max_monomer_extent_A=12.0)

    bad = ok.copy()
    bad[2, 0] = 1.0e8
    assert coords_pathological_for_repack(bad, offsets, max_monomer_extent_A=12.0)

    nan = ok.copy()
    nan[0, 0] = np.nan
    assert coords_pathological_for_repack(nan, offsets, max_monomer_extent_A=12.0)
