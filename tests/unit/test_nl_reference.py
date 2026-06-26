"""Unit tests for nl_reference helpers."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.nl_reference import (
    brute_force_mic_pairs,
    compare_pair_sets,
    filter_pairs_under_cutoff,
    monomer_id_from_offsets,
)


def test_brute_force_two_dimer_pairs() -> None:
    offsets = np.array([0, 3, 6], dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, 6)
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cell = 30.0 * np.eye(3)
    pairs = brute_force_mic_pairs(pos, cell, cutoff=15.0, monomer_id=mid, monomer_offsets=offsets)
    assert len(pairs) > 0
    for ai, aj in pairs:
        assert mid[ai] != mid[aj]
        assert ai < aj


def test_filter_pairs_under_cutoff_drops_shell_pairs() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    cell = 30.0 * np.eye(3)
    raw = {(0, 2), (0, 3)}
    filtered = filter_pairs_under_cutoff(raw, pos, cell, cutoff=6.0)
    assert filtered == {(0, 2)}


def test_compare_pair_sets_symmetric_diff() -> None:
    a = {(0, 1), (2, 3)}
    b = {(0, 1), (4, 5)}
    cmp = compare_pair_sets(a, b)
    assert cmp.only_a == {(2, 3)}
    assert cmp.only_b == {(4, 5)}
    assert not cmp.match
