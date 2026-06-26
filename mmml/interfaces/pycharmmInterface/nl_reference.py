"""Reference neighbor-pair oracles and comparison helpers for MM list validation.

Used by ``tests/functionality/neighbor_lists/`` scripts and ``nl_backend.py``.
Vesin (https://luthaf.fr/vesin/latest/index.html) is the preferred cross-path
reference when installed (``pip install vesin`` or ``uv sync --extra nl-validation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from vesin import NeighborList as VesinNeighborList

    _HAVE_VESIN = True
except ImportError:
    VesinNeighborList = None  # type: ignore[misc, assignment]
    _HAVE_VESIN = False


def have_vesin() -> bool:
    """Return True when the optional ``vesin`` package is importable."""
    return _HAVE_VESIN


def cell_matrix_3x3(cell: np.ndarray) -> np.ndarray:
    """Normalize scalar, (3,), or (3,3) cell spec to a 3×3 matrix (Å)."""
    c = np.asarray(cell, dtype=np.float64)
    if c.ndim == 0:
        L = float(c)
        return np.diag([L, L, L])
    if c.ndim == 1 and c.shape[0] == 3:
        return np.diag(c)
    if c.ndim == 2 and c.shape == (3, 3):
        return c.copy()
    raise ValueError(f"cell must be scalar, (3,), or (3,3); got shape {c.shape}")


def monomer_id_from_offsets(monomer_offsets: Sequence[int], n_atoms: int) -> np.ndarray:
    """Build per-atom monomer index from cumulative offsets."""
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    monomer_id = np.empty(int(n_atoms), dtype=np.int32)
    n_monomers = len(offsets) - 1
    for mi in range(n_monomers):
        monomer_id[int(offsets[mi]) : int(offsets[mi + 1])] = mi
    return monomer_id


def mic_distance(
    positions: np.ndarray,
    ai: int,
    aj: int,
    cell_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two atoms (Å)."""
    dr = positions[aj] - positions[ai]
    inv_cell = np.linalg.inv(cell_matrix)
    frac_dr = dr @ inv_cell.T
    frac_dr = frac_dr - np.round(frac_dr)
    dr_mic = frac_dr @ cell_matrix
    return float(np.linalg.norm(dr_mic))


def extract_valid_pairs(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mask: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    """Return ``{(i, j), ...}`` with ``i < j`` from padded pair arrays."""
    pi = np.asarray(pair_i, dtype=np.int32).reshape(-1)
    pj = np.asarray(pair_j, dtype=np.int32).reshape(-1)
    if mask is None:
        valid = np.ones(pi.shape[0], dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool).reshape(-1)
    out: set[tuple[int, int]] = set()
    for k in range(pi.shape[0]):
        if not valid[k]:
            continue
        i, j = int(pi[k]), int(pj[k])
        if i >= j:
            continue
        out.add((i, j))
    return out


def filter_pairs_under_cutoff(
    pairs: Iterable[tuple[int, int]],
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> set[tuple[int, int]]:
    """Keep pairs with MIC distance strictly below ``cutoff`` (reference contract)."""
    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell)
    cutoff_sq = float(cutoff) ** 2
    out: set[tuple[int, int]] = set()
    for ai, aj in pairs:
        d = mic_distance(R, int(ai), int(aj), cell_mat)
        if d * d < cutoff_sq:
            out.add((int(ai), int(aj)))
    return out


def apply_mm_pair_filters(
    pairs: Iterable[tuple[int, int]],
    *,
    monomer_id: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> set[tuple[int, int]]:
    """Keep inter-monomer pairs; optionally drop pairs with dimer COM distance < mm_r_min."""
    R = np.asarray(positions, dtype=np.float64)
    mid = np.asarray(monomer_id, dtype=np.int32)
    cell_mat = cell_matrix_3x3(cell) if cell is not None else None

    coms: np.ndarray | None = None
    if mm_r_min is not None and monomer_offsets is not None:
        offsets = np.asarray(monomer_offsets, dtype=np.int32)
        n_monomers = len(offsets) - 1
        coms = np.zeros((n_monomers, 3), dtype=np.float64)
        for k in range(n_monomers):
            start, end = int(offsets[k]), int(offsets[k + 1])
            coms[k] = R[start:end].mean(axis=0)

    inv_cell = np.linalg.inv(cell_mat) if cell_mat is not None else None
    mm_r_min_f = float(mm_r_min) if mm_r_min is not None else None

    filtered: set[tuple[int, int]] = set()
    for ai, aj in pairs:
        if int(mid[ai]) == int(mid[aj]):
            continue
        if mm_r_min_f is not None and coms is not None:
            mi, mj = int(mid[ai]), int(mid[aj])
            dr = coms[mj] - coms[mi]
            if inv_cell is not None and cell_mat is not None:
                frac_dr = dr @ inv_cell.T
                frac_dr = frac_dr - np.round(frac_dr)
                dr = frac_dr @ cell_mat
            if float(np.linalg.norm(dr)) < mm_r_min_f:
                continue
        filtered.add((int(ai), int(aj)))
    return filtered


def brute_force_mic_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    *,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> set[tuple[int, int]]:
    """O(N²) MIC reference pairs with ``dist < cutoff`` (matches cell_list contract)."""
    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell)
    inv_cell = np.linalg.inv(cell_mat)
    cutoff_sq = float(cutoff) ** 2
    n = R.shape[0]
    mid = np.asarray(monomer_id, dtype=np.int32)
    raw: set[tuple[int, int]] = set()
    for ai in range(n):
        for aj in range(ai + 1, n):
            if mid[ai] == mid[aj]:
                continue
            dr = R[aj] - R[ai]
            frac_dr = dr @ inv_cell.T
            frac_dr = frac_dr - np.round(frac_dr)
            dr_mic = frac_dr @ cell_mat
            if float(np.dot(dr_mic, dr_mic)) < cutoff_sq:
                raw.add((ai, aj))
    return apply_mm_pair_filters(
        raw,
        monomer_id=mid,
        positions=R,
        cell=cell_mat,
        mm_r_min=mm_r_min,
        monomer_offsets=monomer_offsets,
    )


def vesin_mic_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    *,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> set[tuple[int, int]]:
    """Vesin half-list pairs within ``cutoff``, with MM monomer/COM filters applied."""
    if not _HAVE_VESIN:
        raise ImportError(
            "vesin is not installed. Install with: pip install vesin "
            "or uv sync --extra nl-validation"
        )
    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell)
    calculator = VesinNeighborList(cutoff=float(cutoff), full_list=False)
    i, j, _shifts, dist = calculator.compute(
        points=R,
        box=cell_mat,
        periodic=True,
        quantities="ijSd",
    )
    i = np.asarray(i, dtype=np.int32)
    j = np.asarray(j, dtype=np.int32)
    dist = np.asarray(dist, dtype=np.float64)
    strict = dist < float(cutoff)
    raw = {(int(a), int(b)) for a, b, ok in zip(i, j, strict, strict=False) if ok and a < b}
    return apply_mm_pair_filters(
        raw,
        monomer_id=monomer_id,
        positions=R,
        cell=cell_mat,
        mm_r_min=mm_r_min,
        monomer_offsets=monomer_offsets,
    )


def reference_mic_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    *,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
    prefer_vesin: bool = True,
) -> tuple[set[tuple[int, int]], str]:
    """Return reference pair set and source label (``vesin`` or ``brute``)."""
    if prefer_vesin and _HAVE_VESIN:
        return (
            vesin_mic_pairs(
                positions,
                cell,
                cutoff,
                monomer_id,
                mm_r_min=mm_r_min,
                monomer_offsets=monomer_offsets,
            ),
            "vesin",
        )
    return (
        brute_force_mic_pairs(
            positions,
            cell,
            cutoff,
            monomer_id,
            mm_r_min=mm_r_min,
            monomer_offsets=monomer_offsets,
        ),
        "brute",
    )


@dataclass
class PairSetComparison:
    """Symmetric diff between two neighbor pair sets."""

    only_a: set[tuple[int, int]]
    only_b: set[tuple[int, int]]
    n_a: int
    n_b: int

    @property
    def match(self) -> bool:
        return not self.only_a and not self.only_b

    def summary(self, *, label_a: str = "A", label_b: str = "B", max_show: int = 10) -> str:
        lines = [
            f"{label_a}: {self.n_a} pairs",
            f"{label_b}: {self.n_b} pairs",
            f"only in {label_a}: {len(self.only_a)}",
            f"only in {label_b}: {len(self.only_b)}",
        ]
        if self.only_a:
            sample = sorted(self.only_a)[:max_show]
            lines.append(f"  sample only-{label_a}: {sample}")
        if self.only_b:
            sample = sorted(self.only_b)[:max_show]
            lines.append(f"  sample only-{label_b}: {sample}")
        return "\n".join(lines)


def compare_pair_sets(
    a: Iterable[tuple[int, int]],
    b: Iterable[tuple[int, int]],
) -> PairSetComparison:
    """Compare two half-lists (``i < j``)."""
    set_a = {tuple(p) for p in a}
    set_b = {tuple(p) for p in b}
    return PairSetComparison(
        only_a=set_a - set_b,
        only_b=set_b - set_a,
        n_a=len(set_a),
        n_b=len(set_b),
    )
