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


def test_walk_charmm_primary_jnb_pair_set() -> None:
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        inter_monomer_pair_set,
        walk_charmm_primary_jnb_pair_set,
    )

    # atom1 partners: atom2, atom3; atom2 partner: atom3
    pair_i = [0, 0, 1]
    pair_j = [1, 2, 2]
    raw = walk_charmm_primary_jnb_pair_set(pair_i, pair_j)
    assert raw == {(0, 1), (0, 2), (1, 2)}
    mid = np.array([0, 0, 1, 1], dtype=np.int32)
    inter = inter_monomer_pair_set(raw, monomer_id=mid)
    assert (0, 2) in inter and (1, 2) in inter and (0, 1) not in inter


def test_classify_inter_monomer_diff_cutoff_skew() -> None:
    from mmml.interfaces.pycharmmInterface.nl_reference import classify_inter_monomer_diff

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mid = np.array([0, 0, 1, 1], dtype=np.int32)
    tags = classify_inter_monomer_diff(
        only_left={(0, 2)},
        only_right=set(),
        positions=pos,
        cell=None,
        monomer_id=mid,
        left_cutoff_A=12.0,
        right_cutoff_A=8.0,
        mm_r_min=None,
        monomer_offsets=None,
    )
    assert len(tags["cutoff_only_left"]) == 1
    assert tags["true_mismatch_left"] == []
