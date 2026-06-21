"""Monomer extent / fly-off geometry checks."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import monomer_offsets
from mmml.utils.geometry_checks import (
    assert_monomer_extent_within_limit,
    find_worst_monomer_extent,
)


def test_find_worst_monomer_extent_detects_stretched_monomer():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = monomer_offsets(len(pos), 2)
    extent, violation = find_worst_monomer_extent(pos, offsets)
    assert extent == pytest.approx(10.0498756211)
    assert violation is not None
    assert violation.monomer == 1
    assert violation.atom_start == 3


def test_assert_monomer_extent_within_limit_passes_compact_cluster():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    offsets = monomer_offsets(len(pos), 2)
    worst = assert_monomer_extent_within_limit(
        pos, offsets, max_extent_A=12.0, context="test"
    )
    assert worst == pytest.approx(np.sqrt(2.0))


def test_assert_monomer_extent_raises_on_fly_off():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    offsets = monomer_offsets(len(pos), 2)
    with pytest.raises(RuntimeError, match="monomer extent exceeded"):
        assert_monomer_extent_within_limit(
            pos, offsets, max_extent_A=12.0, context="heat segment"
        )


def test_assert_monomer_extent_raises_on_nonfinite():
    pos = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=float)
    offsets = monomer_offsets(len(pos), 1)
    with pytest.raises(RuntimeError, match="non-finite coordinates"):
        assert_monomer_extent_within_limit(
            pos, offsets, max_extent_A=12.0, context="heat segment"
        )


def test_assert_monomer_extent_unwraps_periodic():
    # Stretched molecule in wrapped coordinates: first atom at 0, second at 29 in a cell of 30.0 Å
    # Actual/unwrapped distance should be 1.0 Å
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [29.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = monomer_offsets(len(pos), 1)
    # Without cell, extent is 29.0, which exceeds 12.0
    with pytest.raises(RuntimeError, match="monomer extent exceeded"):
        assert_monomer_extent_within_limit(
            pos, offsets, max_extent_A=12.0, context="test"
        )
    # With cell, unwrapped extent is 1.0, which is within 12.0
    worst = assert_monomer_extent_within_limit(
        pos, offsets, max_extent_A=12.0, cell=30.0, context="test"
    )
    assert worst == pytest.approx(1.0)

