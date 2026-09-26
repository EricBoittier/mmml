"""Reference neighbor-pair oracles and comparison helpers for MM list validation.

Used by ``tests/functionality/neighbor_lists/`` scripts and ``nl_backend.py``.
Vesin (https://luthaf.fr/vesin/latest/index.html) is the preferred cross-path
reference when installed (``pip install vesin`` or ``uv sync --extra nl-validation``).

Dynamic MM neighbor-list contract:

* Positions passed to rebuild/reference helpers are Cartesian Å coordinates.
  Callers using fractional simulation coordinates must convert them with the
  current cell before entering this layer.
* Cells are scalar, ``(3,)`` orthorhombic lengths, or ``(3, 3)`` matrices in Å.
* Rebuild helpers emit half-list atom pairs with ``i < j`` after MM monomer and
  optional ``mm_r_min`` COM filters.
* Padded arrays use ``int32`` pair indices and boolean masks; valid entries are
  ``mask == True`` and padding entries are ignored regardless of index values.
* Pair order is not part of the API contract. Tests compare pair sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def unique_mic_orthorhombic(cell: np.ndarray, cutoff: float) -> bool:
    """True only when each pair can have at most one image with ``d < cutoff``.

    Uses the strict geometric test ``min(L) > 2*cutoff``. Equality is the
    boundary where two images can sit at ``L/2 = cutoff``; that case must keep
    Vesin's shift list, the distance filter, and sort/dedup.
    """
    cell_mat = cell_matrix_3x3(cell)
    return float(np.min(np.diag(cell_mat))) > 2.0 * float(cutoff)


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


def mm_pair_filter_mask(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    monomer_id: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> np.ndarray:
    """Vectorized MM pair filter: inter-monomer, and dimer COM distance >= ``mm_r_min``.

    Same rule as :func:`apply_mm_pair_filters`; returns a bool mask over the pairs.
    """
    pi = np.asarray(pair_i, dtype=np.int64)
    pj = np.asarray(pair_j, dtype=np.int64)
    mid = np.asarray(monomer_id, dtype=np.int64)
    keep = mid[pi] != mid[pj]
    if mm_r_min is None or monomer_offsets is None or not np.any(keep):
        return keep
    R = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    cell_mat = cell_matrix_3x3(cell) if cell is not None else None
    R = R[: offsets[-1]]
    if cell_mat is not None:
        # Centroids of whole molecules: engines that wrap atoms one at a time (jax-md, ASE wrap())
        # pass molecules split across a face, whose raw centroid would drop switched-on dimers.
        anchor = np.repeat(offsets[:-1], counts)
        frac_a = (R - R[anchor]) @ np.linalg.inv(cell_mat).T
        R = R[anchor] + (frac_a - np.round(frac_a)) @ cell_mat
    coms = np.add.reduceat(R, offsets[:-1], axis=0) / counts[:, None]
    # One COM–COM table (n_mol²) then index by pair monomers — not a MIC
    # per atom pair (n_pairs). ETOH:181 is 181² vs ~6.6e5 pairs.
    dcom = coms[None, :, :] - coms[:, None, :]
    if cell_mat is not None:
        frac = dcom @ np.linalg.inv(cell_mat).T
        dcom = (frac - np.round(frac)) @ cell_mat
    com_ok = np.linalg.norm(dcom, axis=2) >= float(mm_r_min)
    np.fill_diagonal(com_ok, False)
    return keep & com_ok[mid[pi], mid[pj]]


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
    arr = np.asarray(list(pairs), dtype=np.int64).reshape(-1, 2)
    if arr.shape[0] == 0:
        return set()
    keep = mm_pair_filter_mask(
        arr[:, 0],
        arr[:, 1],
        monomer_id=monomer_id,
        positions=positions,
        cell=cell,
        mm_r_min=mm_r_min,
        monomer_offsets=monomer_offsets,
    )
    return set(map(tuple, arr[keep].tolist()))


def canonical_half_pair(ai: int, aj: int) -> tuple[int, int]:
    """Normalize atom pair to ``(i, j)`` with ``i < j``."""
    i, j = int(ai), int(aj)
    return (i, j) if i < j else (j, i)


def walk_charmm_primary_jnb_pair_set(
    pair_i: Sequence[int],
    pair_j: Sequence[int],
) -> set[tuple[int, int]]:
    """Build a half-list set from exported CHARMM JNB ``(i, j)`` arrays."""
    return {
        canonical_half_pair(i, j)
        for i, j in zip(pair_i, pair_j, strict=False)
    }


def walk_charmm_mic_pair_set(
    pair_i: Sequence[int],
    pair_j: Sequence[int],
) -> set[tuple[int, int]]:
    """Build a deduplicated half-list from exported MIC-mapped primary pairs."""
    return walk_charmm_primary_jnb_pair_set(pair_i, pair_j)


def callback_mlmm_pairs_to_half_set(
    idxup: Sequence[int],
    idxvp: Sequence[int],
    *,
    nmlmmp: int,
    natom: int,
) -> set[tuple[int, int]]:
    """Map Fortran ML–MM callback primary indices to a half pair set."""
    n = int(nmlmmp)
    nat = int(natom)
    pairs: set[tuple[int, int]] = set()
    for k in range(n):
        up = int(idxup[k])
        vp = int(idxvp[k])
        if up < 0 or vp < 0 or up >= nat or vp >= nat:
            continue
        pairs.add(canonical_half_pair(up, vp))
    return pairs


def callback_pairs_to_padded_arrays(
    pairs: set[tuple[int, int]],
    *,
    min_capacity: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a half pair set into padded ``(capacity, 2)`` + mask arrays."""
    ordered = sorted(pairs)
    capacity = max(int(min_capacity), len(ordered), 1)
    pair_idx = np.zeros((capacity, 2), dtype=np.int32)
    pair_mask = np.zeros(capacity, dtype=bool)
    for k, (i, j) in enumerate(ordered):
        pair_idx[k, 0] = int(i)
        pair_idx[k, 1] = int(j)
        pair_mask[k] = True
    return pair_idx, pair_mask


def inter_monomer_pair_set(
    pairs: Iterable[tuple[int, int]],
    *,
    monomer_id: np.ndarray,
) -> set[tuple[int, int]]:
    mid = np.asarray(monomer_id, dtype=np.int32)
    return {
        canonical_half_pair(i, j)
        for i, j in pairs
        if int(mid[i]) != int(mid[j])
    }


def classify_inter_monomer_diff(
    *,
    only_left: set[tuple[int, int]],
    only_right: set[tuple[int, int]],
    positions: np.ndarray,
    cell: np.ndarray | None,
    monomer_id: np.ndarray,
    left_cutoff_A: float,
    right_cutoff_A: float,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> dict[str, list[dict[str, float | int]]]:
    """Tag pair mismatches with likely semantic causes (cutoff, COM handoff)."""
    R = np.asarray(positions, dtype=np.float64)
    mid = np.asarray(monomer_id, dtype=np.int32)
    cell_mat = cell_matrix_3x3(cell) if cell is not None else None
    common_cut = min(float(left_cutoff_A), float(right_cutoff_A))

    def _rows(pairs: set[tuple[int, int]], tag: str) -> list[dict[str, float | int]]:
        out: list[dict[str, float | int]] = []
        for i, j in sorted(pairs):
            dist = (
                float(np.linalg.norm(R[j] - R[i]))
                if cell_mat is None
                else mic_distance(R, i, j, cell_mat)
            )
            out.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "distance_A": dist,
                    "monomer_i": int(mid[i]),
                    "monomer_j": int(mid[j]),
                    "tag": tag,
                }
            )
        return out

    cutoff_left = set()
    cutoff_right = set()
    handoff_left = set()
    handoff_right = set()
    true_left = set(only_left)
    true_right = set(only_right)

    for pair in list(only_left):
        i, j = pair
        dist = mic_distance(R, i, j, cell_mat) if cell_mat is not None else float(np.linalg.norm(R[j] - R[i]))
        if dist >= common_cut:
            cutoff_left.add(pair)
            true_left.discard(pair)
    for pair in list(only_right):
        i, j = pair
        dist = mic_distance(R, i, j, cell_mat) if cell_mat is not None else float(np.linalg.norm(R[j] - R[i]))
        if dist >= common_cut:
            cutoff_right.add(pair)
            true_right.discard(pair)

    if mm_r_min is not None and monomer_offsets is not None:
        offsets = np.asarray(monomer_offsets, dtype=np.int32)
        coms = np.zeros((len(offsets) - 1, 3), dtype=np.float64)
        for k in range(len(offsets) - 1):
            s, e = int(offsets[k]), int(offsets[k + 1])
            coms[k] = R[s:e].mean(axis=0)
        inv_cell = np.linalg.inv(cell_mat) if cell_mat is not None else None
        for pair in list(true_left):
            mi, mj = int(mid[pair[0]]), int(mid[pair[1]])
            dr = coms[mj] - coms[mi]
            if inv_cell is not None and cell_mat is not None:
                frac = dr @ inv_cell.T
                frac = frac - np.round(frac)
                dr = frac @ cell_mat
            if float(np.linalg.norm(dr)) < float(mm_r_min):
                handoff_left.add(pair)
                true_left.discard(pair)
        for pair in list(true_right):
            mi, mj = int(mid[pair[0]]), int(mid[pair[1]])
            dr = coms[mj] - coms[mi]
            if inv_cell is not None and cell_mat is not None:
                frac = dr @ inv_cell.T
                frac = frac - np.round(frac)
                dr = frac @ cell_mat
            if float(np.linalg.norm(dr)) < float(mm_r_min):
                handoff_right.add(pair)
                true_right.discard(pair)

    return {
        "cutoff_only_left": _rows(cutoff_left, "cutoff_only"),
        "cutoff_only_right": _rows(cutoff_right, "cutoff_only"),
        "handoff_only_left": _rows(handoff_left, "mm_r_min"),
        "handoff_only_right": _rows(handoff_right, "mm_r_min"),
        "true_mismatch_left": _rows(true_left, "true_mismatch"),
        "true_mismatch_right": _rows(true_right, "true_mismatch"),
    }


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


def vesin_mic_pair_arrays(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    *,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vesin half-list ``(i, j)`` arrays (i < j, lexicographically sorted) after MM filters."""
    if not _HAVE_VESIN:
        raise ImportError(
            "vesin is not installed. Install with: pip install vesin "
            "or uv sync --extra nl-validation"
        )
    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell)
    cutoff = float(cutoff)
    # Unique-MIC (strict L > 2c): skip image-shift fetch. Still filter
    # ``dist < cutoff``, force ``i < j``, and sort/dedup — Vesin does not
    # promise that orientation or order.
    unique_mic = unique_mic_orthorhombic(cell_mat, cutoff)
    calculator = VesinNeighborList(cutoff=cutoff, full_list=False)
    quantities = "ijd" if unique_mic else "ijSd"
    computed = calculator.compute(
        points=R,
        box=cell_mat,
        periodic=True,
        quantities=quantities,
    )
    i = np.asarray(computed[0], dtype=np.int64)
    j = np.asarray(computed[1], dtype=np.int64)
    dist = np.asarray(computed[-1], dtype=np.float64)
    ok = (dist < cutoff) & (i < j)
    i, j = i[ok], j[ok]
    keep = mm_pair_filter_mask(
        i,
        j,
        monomer_id=monomer_id,
        positions=R,
        cell=cell_mat,
        mm_r_min=mm_r_min,
        monomer_offsets=monomer_offsets,
    )
    i, j = i[keep], j[keep]
    key = i * (int(R.shape[0]) + 1) + j
    key = np.sort(key)
    if key.size:
        key = key[np.concatenate(([True], key[1:] != key[:-1]))]
    return key // (int(R.shape[0]) + 1), key % (int(R.shape[0]) + 1)


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
    i, j = vesin_mic_pair_arrays(
        positions, cell, cutoff, monomer_id, mm_r_min=mm_r_min, monomer_offsets=monomer_offsets
    )
    return set(zip(i.tolist(), j.tolist()))


def _array_module(arr):
    """Return ``numpy`` or ``cupy`` module backing ``arr``."""
    type_name = type(arr).__module__
    if type_name.startswith("cupy"):
        import cupy as cp

        return cp
    return np


def vesin_raw_half_list(
    positions,
    cell: np.ndarray,
    cutoff: float,
    *,
    points_module=None,
):
    """Run Vesin half-list on Cartesian Å positions; return ``(i, j, dist)``."""
    if not _HAVE_VESIN:
        raise ImportError("vesin is not installed")
    xp = points_module or _array_module(positions)
    R = xp.asarray(positions, dtype=xp.float64)
    cell_mat = cell_matrix_3x3(np.asarray(cell, dtype=np.float64))
    if xp is not np:
        cell_mat = xp.asarray(cell_mat)
    calculator = VesinNeighborList(cutoff=float(cutoff), full_list=False)
    i, j, _shifts, dist = calculator.compute(
        points=R,
        box=cell_mat,
        periodic=True,
        quantities="ijSd",
    )
    return (
        xp.asarray(i, dtype=xp.int32),
        xp.asarray(j, dtype=xp.int32),
        xp.asarray(dist, dtype=xp.float64),
    )


def filter_vesin_half_list_vectorized(
    i,
    j,
    dist,
    cutoff: float,
    monomer_id,
    positions,
    cell: np.ndarray | None,
    *,
    mm_r_min: float | None = None,
    monomer_offsets: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MM filters on Vesin ``i,j,dist`` (NumPy or CuPy).

    Returns unpadded half-list indices. Pair order follows Vesin/device order and
    should not be treated as stable.
    """
    xp = _array_module(i)
    i_arr = xp.asarray(i, dtype=xp.int32)
    j_arr = xp.asarray(j, dtype=xp.int32)
    dist_arr = xp.asarray(dist, dtype=xp.float64)
    mid = xp.asarray(monomer_id, dtype=xp.int32)
    keep = (dist_arr < float(cutoff)) & (mid[i_arr] != mid[j_arr]) & (i_arr < j_arr)

    if mm_r_min is not None and monomer_offsets is not None:
        R = xp.asarray(positions, dtype=xp.float64)
        offsets = np.asarray(monomer_offsets, dtype=np.int32)
        n_monomers = len(offsets) - 1
        sizes = offsets[1:] - offsets[:-1]
        # Uniform monomers (e.g. solvent boxes): one reshape+mean instead of
        # N tiny device kernels from a Python loop.
        if n_monomers > 0 and int(sizes.min()) == int(sizes.max()):
            apm = int(sizes[0])
            coms = R.reshape(n_monomers, apm, 3).mean(axis=1)
        else:
            coms = xp.zeros((n_monomers, 3), dtype=xp.float64)
            for k in range(n_monomers):
                start, end = int(offsets[k]), int(offsets[k + 1])
                coms[k] = R[start:end].mean(axis=0)
        cell_mat = cell_matrix_3x3(cell) if cell is not None else None
        inv_cell = None
        if cell_mat is not None:
            inv_cell = xp.asarray(np.linalg.inv(cell_mat))
        mi = mid[i_arr]
        mj = mid[j_arr]
        dr = coms[mj] - coms[mi]
        if inv_cell is not None:
            frac_dr = dr @ inv_cell.T
            frac_dr = frac_dr - xp.round(frac_dr)
            dr = frac_dr @ xp.asarray(cell_mat)
        com_dist = xp.linalg.norm(dr, axis=1)
        keep = keep & (com_dist >= float(mm_r_min))

    return i_arr[keep], j_arr[keep]


def pad_pair_arrays(
    pair_i,
    pair_j,
    *,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad filtered pair arrays to fixed capacity (NumPy or CuPy).

    The returned mask is boolean. Entries where ``mask`` is false are padding and
    must be ignored by downstream energy code.
    """
    xp = _array_module(pair_i)
    n_valid = int(pair_i.shape[0])
    if n_valid > max_pairs:
        from mmml.interfaces.pycharmmInterface.cell_list import PairListTruncationError

        raise PairListTruncationError(n_valid, max_pairs)
    cap = int(max_pairs)
    out_i = xp.zeros(cap, dtype=xp.int32)
    out_j = xp.zeros(cap, dtype=xp.int32)
    mask = xp.zeros(cap, dtype=bool)
    out_i[:n_valid] = pair_i
    out_j[:n_valid] = pair_j
    mask[:n_valid] = True
    return out_i, out_j, mask, n_valid


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
