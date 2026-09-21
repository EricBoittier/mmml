"""Deterministic selection, budgets, and D-optimal without explicit inverses."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.acquisition.selection import (
    farthest_point_sampling,
    greedy_doptimal,
    largest_norm_indices,
    stratified_random,
)


def test_fps_is_deterministic_and_respects_budget():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    a = farthest_point_sampling(X, 5, seed=3)
    b = farthest_point_sampling(X, 5, seed=3)
    c = farthest_point_sampling(X, 5, seed=4)
    assert len(a) == 5
    assert len(set(a.tolist())) == 5
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_largest_norm_picks_the_biggest_rows():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 10.0],
            [3.0, 0.0],
            [0.0, 0.1],
        ]
    )
    idx = largest_norm_indices(X, 2)
    assert list(idx) == [1, 2]


def test_stratified_random_covers_strata_and_is_seeded():
    strata = ["A"] * 4 + ["B"] * 4 + ["C"] * 2
    a = stratified_random(strata, 6, seed=0)
    b = stratified_random(strata, 6, seed=0)
    c = stratified_random(strata, 6, seed=1)
    assert len(a) == 6
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    picked = [strata[i] for i in a]
    # round-robin should not starve C
    assert "A" in picked and "B" in picked and "C" in picked


def test_doptimal_prefers_an_orthogonal_direction():
    # Two already-covered axes; the unused axis should be picked first.
    e1 = np.array([[1.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0]])
    e3 = np.array([[0.0, 0.0, 1.0]])
    blocks = [e1, e2, e3]
    idx, gains = greedy_doptimal(blocks, 1, lam=1e-2, seed_blocks=[e1, e2])
    assert idx[0] == 2
    assert gains[0] > 0


def test_doptimal_budget_and_no_repeat():
    rng = np.random.default_rng(2)
    blocks = [rng.normal(size=(2, 4)) for _ in range(8)]
    idx, gains = greedy_doptimal(blocks, 3, lam=1e-3)
    assert len(idx) == 3
    assert len(set(idx.tolist())) == 3
    assert len(gains) == 3


def test_fps_cannot_exceed_pool_size():
    X = np.eye(3)
    idx = farthest_point_sampling(X, 10, seed=0)
    assert len(idx) == 3
