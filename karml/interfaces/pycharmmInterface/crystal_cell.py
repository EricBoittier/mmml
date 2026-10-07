"""CHARMM Nosé–Klein crystal cell.

``XTLAXS`` does not keep the lower-triangular crystallographic embedding.
``GenTen`` builds the metric from ``(a, b, c, α, β, γ)`` and ``XTLAXS`` replaces
it with the unique symmetric positive-definite square root (Nosé and Klein,
Mol. Phys. 1983). An orthogonal cell stays diagonal, so both embeddings match.
A sheared cell has the same metric and a different Cartesian orientation.
Coordinates passed to ``ENER`` have to be written in that symmetric frame.
Placing them in the crystallographic cell rotates them relative to CHARMM's
image lattice, and the shear finite difference scatters.

``GenTen`` snaps an angle to 90° when it lies within ``RSMALL`` degrees.
``number_ltm.F90`` sets ``RSMALL`` to ``1e-4`` in single precision and ``1e-10``
in double precision. This tree's ``chm_real`` is double. A symmetric strain of
``3e-5`` moves γ by about 0.003°, which is outside both thresholds, so the
finite-difference step is not zeroed.
"""

from __future__ import annotations

import numpy as np

# Double-precision ``RSMALL`` in degrees (``GenTen`` / ``XTLAXS``).
CHARMM_ANGLE_RSMALL_DEG = 1.0e-10


def crystallographic_cell(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Lower-triangular cell. Columns are the lattice vectors a, b, c."""
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = float(a)
    bx = float(b) * np.cos(ga)
    by = float(b) * np.sin(ga)
    cx = float(c) * np.cos(be)
    cy = float(c) * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = float(np.sqrt(max(float(c) ** 2 - cx**2 - cy**2, 0.0)))
    return np.array(
        [[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]],
        dtype=np.float64,
    )


def _genten_metric(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Metric tensor ``G_ij = a_i · a_j``, with ``GenTen``'s near-90° snap."""
    al, be, ga = np.radians([alpha, beta, gamma])
    g = np.zeros((3, 3), dtype=np.float64)
    g[0, 0] = float(a) ** 2
    g[1, 1] = float(b) ** 2
    g[2, 2] = float(c) ** 2
    g[1, 2] = g[2, 1] = (
        0.0
        if abs(float(alpha) - 90.0) < CHARMM_ANGLE_RSMALL_DEG
        else float(b) * float(c) * np.cos(al)
    )
    g[0, 2] = g[2, 0] = (
        0.0
        if abs(float(beta) - 90.0) < CHARMM_ANGLE_RSMALL_DEG
        else float(a) * float(c) * np.cos(be)
    )
    g[0, 1] = g[1, 0] = (
        0.0
        if abs(float(gamma) - 90.0) < CHARMM_ANGLE_RSMALL_DEG
        else float(a) * float(b) * np.cos(ga)
    )
    return g


def charmm_symmetric_cell(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Symmetric shape matrix installed by ``XTLAXS``.

    Columns are the lattice vectors. ``H`` is symmetric, so fractional
    coordinates stored as rows transform as ``pos = frac @ H``.
    """
    metric = _genten_metric(a, b, c, alpha, beta, gamma)
    evals, evecs = np.linalg.eigh(metric)
    if np.any(evals <= CHARMM_ANGLE_RSMALL_DEG):
        raise ValueError(
            f"inconsistent cell angles: eigenvalues {evals.tolist()} "
            f"for a,b,c=({a}, {b}, {c}) α,β,γ=({alpha}, {beta}, {gamma})"
        )
    # H = V diag(sqrt(λ)) Vᵀ. Column sign flips cancel. The SPD square root
    # is unique, so the eigenvector order does not matter once eigenvalues
    # are paired with their vectors.
    h = (evecs * np.sqrt(evals)) @ evecs.T
    return 0.5 * (h + h.T)


def cell_metric(cell: np.ndarray) -> np.ndarray:
    """``Hᵀ H`` for a cell whose columns are the lattice vectors."""
    h = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    return h.T @ h
