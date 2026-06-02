from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IntermonomerOverlap:
    """Closest inter-monomer atom-atom contact."""

    monomer_i: int
    monomer_j: int
    atom_i: int
    atom_j: int
    distance_A: float


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


def _mic_displacement(from_pos: np.ndarray, to_pos: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    return _mic(np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float), cell)


def find_worst_intermonomer_overlap(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    cell: Any | None = None,
) -> tuple[float, IntermonomerOverlap | None]:
    """Return the minimum inter-monomer atom distance and the closest pair."""
    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 1:
        return float("inf"), None

    cell_mat = _cell_matrix(cell)
    best_dist = float("inf")
    best: IntermonomerOverlap | None = None

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
                best = IntermonomerOverlap(
                    monomer_i=mi,
                    monomer_j=mj,
                    atom_i=si + int(local_i),
                    atom_j=sj + int(local_j),
                    distance_A=dist,
                )
    return best_dist, best


def push_apart_overlapped_monomers(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    violation: IntermonomerOverlap,
    *,
    min_distance: float,
    margin: float = 0.2,
    cell: Any | None = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Rigidly translate monomer(s) along the MIC vector of the closest contact."""
    pos = np.asarray(positions, dtype=float).copy()
    offsets = np.asarray(monomer_offsets, dtype=int)
    cell_mat = _cell_matrix(cell)
    target = float(min_distance) + float(margin)
    gap = target - float(violation.distance_A)
    if gap <= 0.0:
        return pos

    unit = _mic_displacement(pos[violation.atom_i], pos[violation.atom_j], cell_mat)
    norm = float(np.linalg.norm(unit))
    if norm < 1.0e-12:
        unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        unit = unit / norm

    mi, mj = int(violation.monomer_i), int(violation.monomer_j)
    si, ei = int(offsets[mi]), int(offsets[mi + 1])
    sj, ej = int(offsets[mj]), int(offsets[mj + 1])
    if symmetric:
        pos[si:ei] -= unit * (gap * 0.5)
        pos[sj:ej] += unit * (gap * 0.5)
    else:
        pos[sj:ej] += unit * gap
    return pos


def separate_intermonomer_overlaps(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float,
    margin: float = 0.2,
    cell: Any | None = None,
    max_passes: int | None = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Iteratively push monomer pairs apart until ``min_distance`` is satisfied."""
    pos = np.asarray(positions, dtype=float).copy()
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = max(1, int(len(offsets) - 1))
    passes = int(max_passes) if max_passes is not None else max(1, n_monomers * 2)
    threshold = float(min_distance)

    for _ in range(passes):
        best_dist, violation = find_worst_intermonomer_overlap(pos, offsets, cell=cell)
        if violation is None or best_dist >= threshold:
            break
        pos = push_apart_overlapped_monomers(
            pos,
            offsets,
            violation,
            min_distance=threshold,
            margin=margin,
            cell=cell,
            symmetric=symmetric,
        )
    return pos


def assert_no_intermonomer_atom_overlap(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float = 0.1,
    cell: Any | None = None,
    context: str = "geometry",
) -> float:
    """Raise if atoms from different monomers are closer than min_distance.

    Returns the minimum inter-monomer atom distance found.
    """
    threshold = float(min_distance)
    if threshold <= 0.0:
        return float("inf")

    best_dist, violation = find_worst_intermonomer_overlap(
        positions, monomer_offsets, cell=cell
    )
    if violation is not None and best_dist < threshold:
        raise RuntimeError(
            f"{context}: inter-monomer atom overlap detected: "
            f"monomers {violation.monomer_i}/{violation.monomer_j}, "
            f"atoms {violation.atom_i}/{violation.atom_j}, distance={best_dist:.4f} A "
            f"< min_distance={threshold:.4f} A"
        )
    return best_dist
