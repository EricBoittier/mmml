"""Batch selection: stratified random, FPS, largest-norm, greedy D-optimal.

Distance / magnitude convention
-------------------------------
Diversity methods use **Euclidean distance in PCA embedding space**.  PCA
centers and (by default) standard-scales each coordinate of the
representation.  Representation rows are **not** L2-normalized before PCA
unless ``row_normalize=True``, so vector magnitude can still influence the
embedding.  The largest-norm ablation instead ranks the raw (pre-PCA) L2
norm of the representation.  That tests whether “lighting up most” helps
relative to diversity-aware selection.

This is **not** the original classification BADGE algorithm (which clusters
gradient embeddings for a classification loss).  We only reuse the idea that
a per-example gradient (or activation) can be a representation for
diversity sampling, with explicit magnitude handling as above.

D-optimal information gain never forms a matrix inverse: gains use a
Cholesky factor of the regularized Gram matrix ``A`` and either the
matrix-determinant lemma (via ``cho_solve``) or a fresh Cholesky of
``A + J^T J``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy import linalg as sla


def farthest_point_sampling(
    X: np.ndarray,
    n_select: int,
    *,
    seed: int = 0,
    init_index: int | None = None,
    selected: Sequence[int] | None = None,
) -> np.ndarray:
    """Greedy k-center / farthest-point sampling in Euclidean space.

    Starts from ``init_index`` if given, else a seeded random point (or the
    provided ``selected`` prefix).  Subsequent points maximize the minimum
    distance to the already chosen set.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n = X.shape[0]
    k = min(int(n_select), n)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    chosen: list[int] = [int(i) for i in (selected or ())]
    remaining = set(range(n)) - set(chosen)
    rng = np.random.default_rng(int(seed))
    if not chosen:
        if init_index is not None:
            start = int(init_index)
        else:
            start = int(rng.integers(0, n))
        chosen.append(start)
        remaining.discard(start)
    if len(chosen) >= k:
        return np.asarray(chosen[:k], dtype=np.int64)
    # min-distance to the chosen set
    dist = np.full(n, np.inf, dtype=np.float64)
    for j in chosen:
        diff = X - X[j]
        dist = np.minimum(dist, np.einsum("ij,ij->i", diff, diff))
    while len(chosen) < k:
        # already chosen have dist 0 after update; mask them
        masked = dist.copy()
        masked[np.array(chosen, dtype=np.int64)] = -np.inf
        nxt = int(np.argmax(masked))
        chosen.append(nxt)
        diff = X - X[nxt]
        dist = np.minimum(dist, np.einsum("ij,ij->i", diff, diff))
    return np.asarray(chosen, dtype=np.int64)


def largest_norm_indices(X: np.ndarray, n_select: int) -> np.ndarray:
    """Select the ``n_select`` rows with largest L2 norm (diagnostic ablation)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    k = min(int(n_select), X.shape[0])
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    norms = np.linalg.norm(X, axis=1)
    # argsort descending; np.argpartition then sort for determinism
    idx = np.argpartition(-norms, kth=k - 1)[:k]
    idx = idx[np.argsort(-norms[idx], kind="stable")]
    return np.asarray(idx, dtype=np.int64)


def stratified_random(
    strata: Sequence[str],
    n_select: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Sample randomly within composition × condition strata.

    Budget is spread as evenly as possible across non-empty strata; leftovers
    go to the largest strata.  Sampling inside a stratum is without
    replacement.  Deterministic given ``seed``.
    """
    strata_arr = np.asarray(list(strata), dtype=object)
    n = len(strata_arr)
    k = min(int(n_select), n)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(strata_arr.tolist()):
        buckets[str(s)].append(i)
    rng = np.random.default_rng(int(seed))
    keys = sorted(buckets)
    # shuffle indices inside each stratum with a per-stratum stream
    shuffled: dict[str, list[int]] = {}
    for j, key in enumerate(keys):
        local = np.random.default_rng(rng.integers(0, 2**31 - 1) + j)
        idx = np.asarray(buckets[key], dtype=np.int64)
        shuffled[key] = local.permutation(idx).tolist()
    # round-robin so small strata are not starved
    chosen: list[int] = []
    cursors = {key: 0 for key in keys}
    while len(chosen) < k:
        progress = False
        for key in keys:
            c = cursors[key]
            pool = shuffled[key]
            if c < len(pool):
                chosen.append(int(pool[c]))
                cursors[key] = c + 1
                progress = True
                if len(chosen) >= k:
                    break
        if not progress:
            break
    return np.asarray(chosen, dtype=np.int64)


def _logdet_chol(A: np.ndarray) -> float:
    c, lower = sla.cho_factor(A, lower=True, check_finite=False)
    diag = np.diag(c) if c.ndim == 2 else c
    # cho_factor overwrites with L; logdet(A) = 2 sum log L_ii
    return float(2.0 * np.log(np.abs(diag)).sum())


def _gain_matrix_det_lemma(chol: tuple[np.ndarray, bool], J: np.ndarray) -> float:
    """logdet(I + J A^{-1} J^T) via cho_solve (no explicit inverse)."""
    m, p = J.shape
    if m == 0:
        return 0.0
    # A^{-1} J^T
    x = sla.cho_solve(chol, J.T, check_finite=False)
    mmat = np.eye(m, dtype=np.float64) + J @ x
    mmat = 0.5 * (mmat + mmat.T)
    sign, logdet = np.linalg.slogdet(mmat)
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def greedy_doptimal(
    blocks: Sequence[np.ndarray],
    n_select: int,
    *,
    lam: float = 1e-3,
    seed_blocks: Sequence[np.ndarray] | None = None,
    already_selected: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Greedy regularized D-optimal selection on per-structure Jacobian blocks.

    ``blocks[i]`` is ``J_x`` with shape ``(n_obs_i, n_params)``.  Initialize

    ``A = λ I + sum_{x in seed} J_x^T J_x``

    (plus any ``already_selected`` blocks) and at each step pick ``x``
    maximizing ``logdet(A + J^T J) - logdet A``, then update ``A``.
    """
    if not blocks:
        return np.zeros((0,), dtype=np.int64), []
    p = int(blocks[0].shape[1])
    if p < 1:
        raise ValueError("Jacobian blocks have empty parameter axis")
    A = float(lam) * np.eye(p, dtype=np.float64)
    if seed_blocks:
        for J in seed_blocks:
            J = np.asarray(J, dtype=np.float64)
            if J.size:
                A = A + J.T @ J
    chosen: list[int] = [int(i) for i in (already_selected or ())]
    remaining = set(range(len(blocks))) - set(chosen)
    for i in chosen:
        J = np.asarray(blocks[i], dtype=np.float64)
        A = A + J.T @ J
    k = min(int(n_select), len(blocks))
    gains: list[float] = []
    while len(chosen) < k and remaining:
        chol = sla.cho_factor(A, lower=True, check_finite=False)
        best_i = None
        best_gain = float("-inf")
        for i in remaining:
            J = np.asarray(blocks[i], dtype=np.float64)
            if J.size == 0:
                continue
            m = J.shape[0]
            if m <= p:
                gain = _gain_matrix_det_lemma(chol, J)
            else:
                A_new = A + J.T @ J
                gain = _logdet_chol(A_new) - _logdet_chol(A)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        chosen.append(int(best_i))
        remaining.remove(best_i)
        J = np.asarray(blocks[best_i], dtype=np.float64)
        A = A + J.T @ J
        gains.append(float(best_gain))
    return np.asarray(chosen, dtype=np.int64), gains


def maybe_row_normalize(X: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.asarray(X, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    return X / nrm
