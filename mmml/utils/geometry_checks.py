from __future__ import annotations

from typing import Any

import numpy as np


def _cell_matrix(cell: Any | None) -> np.ndarray | None:
    if cell is None:
        return None
    arr = np.asarray(cell, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        if value <= 0.0:
            return None
        return np.diag([value, value, value])
    if arr.shape == (1,):
        value = float(arr[0])
        if value <= 0.0:
            return None
        return np.diag([value, value, value])
    if arr.shape == (3,):
        if np.any(arr <= 0.0):
            return None
        return np.diag(arr)
    if arr.shape == (3, 3):
        if abs(float(np.linalg.det(arr))) < 1.0e-12:
            return None
        return arr
    return None


def _mic(displacements: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    if cell is None:
        return displacements
    inv_cell = np.linalg.inv(cell)
    frac = displacements @ inv_cell.T
    frac = frac - np.round(frac)
    return frac @ cell


def assert_no_intermonomer_atom_overlap(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float = 0.5,
    cell: Any | None = None,
    context: str = "geometry",
) -> float:
    """Raise if atoms from different monomers are closer than min_distance.

    Returns the minimum inter-monomer atom distance found.
    """
    threshold = float(min_distance)
    if threshold <= 0.0:
        return float("inf")

    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 1:
        return float("inf")

    cell_mat = _cell_matrix(cell)
    best_dist = float("inf")
    best: tuple[int, int, int, int] | None = None

    for mi in range(n_monomers):
        si, ei = int(offsets[mi]), int(offsets[mi + 1])
        ri = pos[si:ei]
        for mj in range(mi + 1, n_monomers):
            sj, ej = int(offsets[mj]), int(offsets[mj + 1])
            rj = pos[sj:ej]
            disp = _mic(ri[:, None, :] - rj[None, :, :], cell_mat)
            d2 = np.sum(disp * disp, axis=-1)
            flat_idx = int(np.argmin(d2))
            local_i, local_j = np.unravel_index(flat_idx, d2.shape)
            dist = float(np.sqrt(d2[local_i, local_j]))
            if dist < best_dist:
                best_dist = dist
                best = (mi, mj, si + int(local_i), sj + int(local_j))

    if best is not None and best_dist < threshold:
        mi, mj, ai, aj = best
        raise RuntimeError(
            f"{context}: inter-monomer atom overlap detected: "
            f"monomers {mi}/{mj}, atoms {ai}/{aj}, distance={best_dist:.4f} A "
            f"< min_distance={threshold:.4f} A"
        )
    return best_dist
