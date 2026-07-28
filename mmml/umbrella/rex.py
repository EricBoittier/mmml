"""Hamiltonian replica exchange for packed umbrella windows.

Only the harmonic biases differ between windows, so the Metropolis criterion
reduces to bias energies:

    Δ = W_a(R_b) + W_b(R_a) − W_a(R_a) − W_b(R_b)
    P_acc = min(1, exp(−β Δ))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RexStats:
    """Running replica-exchange acceptance counters."""

    attempted: int = 0
    accepted: int = 0

    @property
    def acceptance(self) -> float:
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted


def neighbor_exchange_pairs(
    grid_shape: Sequence[int],
    phase: int,
) -> list[tuple[int, int]]:
    """Even/odd neighbor pairs on a 1D chain or 2D product grid.

    Window indexing matches ``np.meshgrid(..., indexing="ij").ravel()``:
    ``index = ix * ny + iy`` for shape ``(nx, ny)``.
    """
    shape = tuple(int(x) for x in grid_shape)
    if len(shape) == 1:
        n = shape[0]
        start = int(phase) % 2
        return [(i, i + 1) for i in range(start, n - 1, 2)]
    if len(shape) != 2:
        raise ValueError(f"only 1D/2D grids supported for RE (got shape={shape})")
    nx, ny = shape
    phase = int(phase) % 4

    def idx(ix: int, iy: int) -> int:
        return ix * ny + iy

    pairs: list[tuple[int, int]] = []
    if phase in (0, 1):
        parity = phase % 2
        for ix in range(nx):
            for iy in range(parity, ny - 1, 2):
                pairs.append((idx(ix, iy), idx(ix, iy + 1)))
    else:
        parity = phase % 2
        for iy in range(ny):
            for ix in range(parity, nx - 1, 2):
                pairs.append((idx(ix, iy), idx(ix + 1, iy)))
    return pairs


def bias_energy_matrix(
    cv: np.ndarray,
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
) -> np.ndarray:
    """``W[i, j]`` = window-``i`` bias evaluated on configuration ``j``.

    ``cv`` has shape ``(K, ndim)``; ``targets_per_cv`` / ``k_per_cv`` are
    ``(ndim, K)``-like.
    """
    cv_arr = np.asarray(cv, dtype=np.float64)
    if cv_arr.ndim != 2:
        raise ValueError(f"cv must have shape (K, ndim), got {cv_arr.shape}")
    k_windows, ndim = cv_arr.shape
    if len(targets_per_cv) != ndim or len(k_per_cv) != ndim:
        raise ValueError("targets_per_cv / k_per_cv length must match cv.ndim")
    w = np.zeros((k_windows, k_windows), dtype=np.float64)
    for d in range(ndim):
        targets = np.asarray(targets_per_cv[d], dtype=np.float64)
        ks = np.asarray(k_per_cv[d], dtype=np.float64)
        if targets.shape != (k_windows,) or ks.shape != (k_windows,):
            raise ValueError(
                f"CV {d}: targets/k must have length K={k_windows}, "
                f"got {targets.shape}/{ks.shape}"
            )
        # (i, j): window i restraints on config j
        diff = cv_arr[None, :, d] - targets[:, None]
        w += 0.5 * ks[:, None] * np.square(diff)
    return w


def metropolis_exchange_delta(w_matrix: np.ndarray, a: int, b: int) -> float:
    """Bias-only Δ for swapping configurations between windows ``a`` and ``b``."""
    return float(
        w_matrix[a, b] + w_matrix[b, a] - w_matrix[a, a] - w_matrix[b, b]
    )


def attempt_replica_exchanges(
    *,
    positions_packed: np.ndarray,
    momenta_packed: np.ndarray | None,
    forces_packed: np.ndarray | None,
    cv: np.ndarray,
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
    grid_shape: Sequence[int],
    phase: int,
    beta: float,
    rng: np.random.Generator,
    n_atoms: int,
    stats: RexStats | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    """Try even/odd neighbor swaps; return updated packed arrays and counts.

    Swaps configurations (and momenta/forces when provided) between window
    slots. Restraints stay with their window indices.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0 (got {beta})")
    pos = np.asarray(positions_packed, dtype=np.float64).reshape(-1, n_atoms, 3)
    k_windows = int(pos.shape[0])
    mom = None
    frc = None
    if momenta_packed is not None:
        mom = np.asarray(momenta_packed, dtype=np.float64).reshape(k_windows, n_atoms, 3)
    if forces_packed is not None:
        frc = np.asarray(forces_packed, dtype=np.float64).reshape(k_windows, n_atoms, 3)

    w = bias_energy_matrix(cv, targets_per_cv, k_per_cv)
    pairs = neighbor_exchange_pairs(grid_shape, phase)
    attempted = 0
    accepted = 0
    for a, b in pairs:
        if a < 0 or b >= k_windows or a >= b:
            continue
        attempted += 1
        delta = metropolis_exchange_delta(w, a, b)
        log_acc = -float(beta) * delta
        if log_acc >= 0.0 or rng.random() < float(np.exp(min(log_acc, 0.0))):
            accepted += 1
            pos[[a, b]] = pos[[b, a]]
            if mom is not None:
                mom[[a, b]] = mom[[b, a]]
            if frc is not None:
                frc[[a, b]] = frc[[b, a]]
            # Configs moved between slots a↔b; W[i,j] is bias_i on config_j
            w[:, [a, b]] = w[:, [b, a]]

    if stats is not None:
        stats.attempted += attempted
        stats.accepted += accepted

    pos_out = pos.reshape(k_windows * n_atoms, 3)
    mom_out = None if mom is None else mom.reshape(k_windows * n_atoms, 3)
    frc_out = None if frc is None else frc.reshape(k_windows * n_atoms, 3)
    return pos_out, mom_out, frc_out, attempted, accepted
