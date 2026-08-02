"""Tests for host MIC pair construction used by hybrid MD neighbor refresh."""

from __future__ import annotations

import time

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    _build_pair_indices,
    _build_pair_indices_vectorized,
)


def _naive_pairs(pos, cell, excluded, cutoff):
    pos = np.asarray(pos, dtype=np.float64)
    cell_mat = np.asarray(cell, dtype=np.float64)
    if cell_mat.shape == (3,):
        cell_mat = np.diag(cell_mat)
    inv = np.linalg.inv(cell_mat)
    cutoff_sq = float(cutoff) ** 2
    n = pos.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            dr = pos[j] - pos[i]
            frac = dr @ inv.T
            frac = frac - np.round(frac)
            dr_mic = frac @ cell_mat
            if float(np.dot(dr_mic, dr_mic)) < cutoff_sq:
                pairs.add((i, j))
    return pairs


def _pair_set(pi, pj):
    return {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in zip(pi, pj)}


def test_build_pair_indices_matches_naive_small():
    rng = np.random.default_rng(0)
    n = 40
    box = 18.0
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = np.diag([box, box, box])
    excluded = frozenset({(0, 1), (2, 5), (10, 11)})
    cutoff = 6.0

    pi, pj = _build_pair_indices(pos, cell, excluded, cutoff)
    assert _pair_set(pi, pj) == _naive_pairs(pos, cell, excluded, cutoff)


def test_build_pair_indices_large_cutoff_uses_mic_unique_pairs():
    """When cutoff > L/2, Vesin images must not inflate the MIC pair set."""
    rng = np.random.default_rng(3)
    n = 12
    box = 10.0
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = np.diag([box, box, box])
    excluded = frozenset({(0, 2)})
    cutoff = 9.0  # > L/2

    pi, pj = _build_pair_indices(pos, cell, excluded, cutoff)
    assert _pair_set(pi, pj) == _naive_pairs(pos, cell, excluded, cutoff)


def test_vectorized_fallback_matches_naive():
    rng = np.random.default_rng(1)
    n = 35
    box = 16.0
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = np.diag([box, box, box])
    excluded = frozenset({(1, 2), (3, 7)})
    cutoff = 5.5

    pi, pj = _build_pair_indices_vectorized(pos, cell, excluded, cutoff)
    assert _pair_set(pi, pj) == _naive_pairs(pos, cell, excluded, cutoff)


def test_build_pair_indices_scales_for_solvent_box():
    """~2k atoms must not take minutes in a pure-Python O(N²) loop."""
    rng = np.random.default_rng(2)
    n = 900
    box = 25.0
    pos = rng.uniform(0.0, box, size=(n, 3))
    cell = np.diag([box, box, box])
    excluded: frozenset[tuple[int, int]] = frozenset()
    cutoff = 8.0

    t0 = time.perf_counter()
    pi, pj = _build_pair_indices(pos, cell, excluded, cutoff)
    elapsed = time.perf_counter() - t0

    assert pi.size == pj.size
    assert pi.size > 0
    assert elapsed < 2.0, f"pair build too slow: {elapsed:.2f}s for N={n}"
