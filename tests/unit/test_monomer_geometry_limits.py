"""Unit tests for bond-derived monomer geometry limits."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.monomer_geometry_limits import (
    DEFAULT_INTRA_MIN_DISTANCE_A,
    DEFAULT_MAX_MONOMER_EXTENT_A,
    compute_monomer_geometry_limits,
)


def test_compute_limits_tighter_than_legacy_defaults() -> None:
    # Linear 4-atom chain along x: bonds ~1.5 Å, extent ~4.5 Å
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 4], dtype=int)
    bonds = [(0, 1), (1, 2), (2, 3)]
    limits = compute_monomer_geometry_limits(
        pos,
        offsets,
        bond_pairs_12=bonds,
        excluded_pairs=frozenset(bonds),
    )
    assert limits is not None
    assert limits.max_monomer_extent_A < DEFAULT_MAX_MONOMER_EXTENT_A
    assert limits.intra_min_distance_A > DEFAULT_INTRA_MIN_DISTANCE_A
    assert limits.max_bond_length_A == 1.5
    assert limits.reference_max_extent_A == 4.5


def test_compute_limits_two_monomers_use_worst_case() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.5, 0.0, 0.0],
            [12.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 5], dtype=int)
    bonds = [(0, 1), (2, 3), (3, 4)]
    limits = compute_monomer_geometry_limits(
        pos,
        offsets,
        bond_pairs_12=bonds,
        excluded_pairs=frozenset(bonds),
    )
    assert limits is not None
    assert limits.reference_max_extent_A == 2.5
    assert limits.max_monomer_extent_A < 8.0
