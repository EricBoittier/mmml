from __future__ import annotations

import numpy as np
import pytest

from scripts.reorder_dimer_atoms_to_reference import (
    _reorder_per_atom_extras,
    match_permutation,
)


def test_match_permutation_recovers_shuffled_dimer_order() -> None:
    reference_numbers = np.array([6, 17, 17, 1, 1, 6, 17, 17, 1, 1], dtype=np.int32)
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.7, 0.0, 0.0],
            [-1.7, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.7, 0.0, 0.0],
            [3.3, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    shuffled_to_reference = np.array([5, 7, 6, 9, 8, 0, 2, 1, 4, 3], dtype=int)
    source_positions = reference_positions[shuffled_to_reference]
    source_numbers = reference_numbers[shuffled_to_reference]

    perm, rmsd = match_permutation(
        source_positions,
        source_numbers,
        reference_positions,
        reference_numbers,
    )

    assert source_numbers[perm].tolist() == reference_numbers.tolist()
    assert source_positions[perm] == pytest.approx(reference_positions)
    assert rmsd == pytest.approx(0.0, abs=1e-12)


def test_match_permutation_rejects_wrong_formula() -> None:
    reference_positions = np.zeros((2, 3), dtype=np.float64)
    source_positions = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="atomic-number multisets differ"):
        match_permutation(
            source_positions,
            np.array([6, 1], dtype=np.int32),
            reference_positions,
            np.array([6, 17], dtype=np.int32),
        )


def test_reorder_per_atom_extras_respects_padding() -> None:
    forces = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    dipoles = np.arange(2 * 3, dtype=np.float64).reshape(2, 3)
    extras = {"F": forces, "D": dipoles, "N": np.array([3, 5])}
    permutations = [
        np.array([2, 0, 1], dtype=int),
        np.array([4, 3, 2, 1, 0], dtype=int),
    ]

    reordered = _reorder_per_atom_extras(
        extras,
        permutations=permutations,
        active_counts=np.array([3, 5]),
        max_atoms=5,
    )

    assert reordered["F"][0, :3] == pytest.approx(forces[0, [2, 0, 1]])
    assert reordered["F"][0, 3:] == pytest.approx(forces[0, 3:])
    assert reordered["F"][1] == pytest.approx(forces[1, [4, 3, 2, 1, 0]])
    assert reordered["D"] is dipoles
