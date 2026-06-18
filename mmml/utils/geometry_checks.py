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


@dataclass(frozen=True)
class IntramonomerCloseContact:
    """Closest nonbonded atom-atom contact within one monomer."""

    monomer: int
    atom_i: int
    atom_j: int
    distance_A: float


@dataclass(frozen=True)
class MonomerExtentViolation:
    """Monomer whose axis-aligned bounding box exceeds the allowed span."""

    monomer: int
    extent_A: float
    atom_start: int


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


def wrap_monomers_primary_cell(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    cell: Any,
) -> np.ndarray:
    """Rigid integer-lattice shifts per monomer so each COM sits in the primary orthorhombic cell."""
    pos = np.asarray(positions, dtype=float).copy()
    cell_mat = _cell_matrix(cell)
    if cell_mat is None:
        return pos
    Lx, Ly, Lz = float(cell_mat[0, 0]), float(cell_mat[1, 1]), float(cell_mat[2, 2])
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_mol = int(len(offsets) - 1)
    for mi in range(n_mol):
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        com = pos[s:e].mean(axis=0)
        shift = np.array(
            [
                -np.floor(com[0] / Lx) * Lx if Lx > 0 else 0.0,
                -np.floor(com[1] / Ly) * Ly if Ly > 0 else 0.0,
                -np.floor(com[2] / Lz) * Lz if Lz > 0 else 0.0,
            ],
            dtype=float,
        )
        pos[s:e] += shift
    return pos


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
        )
    return best_dist


def normalize_atom_pair(i: int, j: int) -> tuple[int, int]:
    """Return ``(min, max)`` for an unordered atom pair."""
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)


def build_bond_exclusion_pairs(
    ib: Any,
    jb: Any,
    *,
    exclude_1_3: bool = True,
) -> frozenset[tuple[int, int]]:
    """Return 1–2 (and optionally 1–3) atom pairs to skip in close-contact scans.

    ``ib`` / ``jb`` follow CHARMM PSF convention (1-based atom indices).
    """
    bonds: set[tuple[int, int]] = set()
    neighbors: dict[int, set[int]] = {}
    for i_raw, j_raw in zip(ib, jb):
        a, b = int(i_raw) - 1, int(j_raw) - 1
        if a < 0 or b < 0:
            continue
        pair = normalize_atom_pair(a, b)
        bonds.add(pair)
        neighbors.setdefault(pair[0], set()).add(pair[1])
        neighbors.setdefault(pair[1], set()).add(pair[0])

    excluded = set(bonds)
    if exclude_1_3:
        for a, b in bonds:
            for c in neighbors.get(a, ()):
                if c != b:
                    excluded.add(normalize_atom_pair(b, c))
            for c in neighbors.get(b, ()):
                if c != a:
                    excluded.add(normalize_atom_pair(a, c))
    return frozenset(excluded)


def find_worst_intramonomer_close_contact(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    excluded_pairs: frozenset[tuple[int, int]] | set[tuple[int, int]],
    *,
    cell: Any | None = None,
) -> tuple[float, IntramonomerCloseContact | None]:
    """Return the minimum non-excluded atom distance inside any monomer."""
    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 0:
        return float("inf"), None

    cell_mat = _cell_matrix(cell)
    excluded = frozenset(excluded_pairs)
    best_dist = float("inf")
    best: IntramonomerCloseContact | None = None

    for mi in range(n_monomers):
        si, ei = int(offsets[mi]), int(offsets[mi + 1])
        n_local = ei - si
        if n_local < 2:
            continue
        block = pos[si:ei]
        for local_i in range(n_local):
            for local_j in range(local_i + 1, n_local):
                gi = si + local_i
                gj = si + local_j
                if normalize_atom_pair(gi, gj) in excluded:
                    continue
                disp = _mic_displacement(block[local_i], block[local_j], cell_mat)
                dist = float(np.linalg.norm(disp))
                if dist < best_dist:
                    best_dist = dist
                    best = IntramonomerCloseContact(
                        monomer=mi,
                        atom_i=gi,
                        atom_j=gj,
                        distance_A=dist,
                    )
    return best_dist, best


def monomer_axis_extent(positions: np.ndarray, monomer_offsets: np.ndarray, monomer: int) -> float:
    """Axis-aligned bounding-box span for one monomer (Å)."""
    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    mi = int(monomer)
    si, ei = int(offsets[mi]), int(offsets[mi + 1])
    if ei <= si:
        return 0.0
    chunk = pos[si:ei]
    return float(np.linalg.norm(chunk.max(axis=0) - chunk.min(axis=0)))


def find_worst_monomer_extent(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
) -> tuple[float, MonomerExtentViolation | None]:
    """Return the largest monomer axis extent and the worst offender."""
    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 0:
        return 0.0, None

    worst_extent = 0.0
    worst: MonomerExtentViolation | None = None
    for mi in range(n_monomers):
        extent = monomer_axis_extent(pos, offsets, mi)
        if extent > worst_extent:
            worst_extent = extent
            worst = MonomerExtentViolation(
                monomer=mi,
                extent_A=extent,
                atom_start=int(offsets[mi]),
            )
    return worst_extent, worst


def assert_monomer_extent_within_limit(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    max_extent_A: float,
    context: str = "geometry",
) -> float:
    """Raise if any monomer span exceeds ``max_extent_A`` or coords are non-finite."""
    pos = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(pos)):
        n_bad = int(np.sum(~np.isfinite(pos)))
        raise RuntimeError(
            f"{context}: non-finite coordinates detected ({n_bad} atom(s))"
        )

    limit = float(max_extent_A)
    if limit <= 0.0:
        worst_extent, _ = find_worst_monomer_extent(pos, monomer_offsets)
        return float(worst_extent)

    worst_extent, violation = find_worst_monomer_extent(pos, monomer_offsets)
    if violation is not None and worst_extent > limit:
        raise RuntimeError(
            f"{context}: monomer extent exceeded: "
            f"monomer {violation.monomer + 1} (atom {violation.atom_start + 1}+) "
            f"extent={worst_extent:.2f} A > max={limit:.2f} A"
        )
    return float(worst_extent)


def assert_no_intramonomer_close_contact(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    excluded_pairs: frozenset[tuple[int, int]] | set[tuple[int, int]],
    *,
    min_distance: float = 1.0,
    cell: Any | None = None,
    context: str = "geometry",
) -> float:
    """Raise if any monomer has a non-excluded atom pair closer than ``min_distance``."""
    threshold = float(min_distance)
    if threshold <= 0.0:
        return float("inf")

    best_dist, violation = find_worst_intramonomer_close_contact(
        positions,
        monomer_offsets,
        excluded_pairs,
        cell=cell,
    )
    if violation is not None and best_dist < threshold:
        raise RuntimeError(
            f"{context}: intra-monomer close contact: "
            f"monomer {violation.monomer}, atoms {violation.atom_i}/{violation.atom_j}, "
            f"distance={best_dist:.4f} A < min_distance={threshold:.4f} A"
        )
    return best_dist
