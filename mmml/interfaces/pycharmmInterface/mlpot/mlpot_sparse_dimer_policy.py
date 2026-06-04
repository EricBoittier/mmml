"""Sparse ML dimer slot cap and near-neighbor counting (no JAX import)."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import dimer_permutations


def max_dimer_pairs(n_monomers: int) -> int:
    """Maximum number of unique monomer dimers for an ``n_monomers`` cluster."""
    n = int(n_monomers)
    return max(0, n * (n - 1) // 2)


def resolve_max_active_dimers(
    n_monomers: int,
    n_dimers_total: Optional[int] = None,
    explicit: Optional[int] = None,
    *,
    free_space: bool = False,
) -> int:
    """Max PhysNet dimer slots per step when ``ml_sparse_dimers`` is enabled.

    Periodic systems default to ``max(1000, 6n)`` because the active-pair
    count is bounded by local density.  Free-space clusters have no comparable
    geometric bound: the largest safe static allocation is every unique dimer,
    ``n * (n - 1) // 2`` (DCM:90 -> 4005 slots).  Lower explicit/env caps are
    promoted in free-space mode so sparse packing cannot silently drop pairs.
    """
    if n_dimers_total is None:
        n_dimers_total = max_dimer_pairs(n_monomers)
    n_dimers_total = int(n_dimers_total)

    all_pairs_cap = n_dimers_total
    if explicit is not None:
        cap = int(explicit)
    else:
        env = (os.environ.get("MMML_MLPOT_MAX_ACTIVE_DIMERS") or "").strip()
        if env:
            cap = int(env)
        else:
            cap = all_pairs_cap if free_space else max(1000, 6 * int(n_monomers))
    if free_space:
        cap = max(cap, all_pairs_cap)
    cap = min(all_pairs_cap, cap)
    return max(1, cap)


def build_monomer_dimer_index_arrays(
    n_monomers: int,
    atoms_per_monomer: int | Sequence[int],
    *,
    max_atoms: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build padded monomer/dimer index arrays matching ``setup_calculator``."""
    if isinstance(atoms_per_monomer, int):
        per = [int(atoms_per_monomer)] * int(n_monomers)
    else:
        per = [int(x) for x in atoms_per_monomer]
    if len(per) != int(n_monomers):
        raise ValueError(f"atoms_per_monomer length {len(per)} != n_monomers {n_monomers}")

    offsets = np.zeros(int(n_monomers) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(per)
    n_atoms = int(offsets[-1])
    max_m = max(per)
    max_d = max(a + b for a, b in ((per[i], per[j]) for i, j in dimer_permutations(n_monomers)))
    if max_atoms is None:
        max_atoms = max(max_m, max_d)

    monomer_idx = np.zeros((int(n_monomers), max_atoms), dtype=np.int32)
    for mi in range(int(n_monomers)):
        idxs = np.arange(offsets[mi], offsets[mi + 1], dtype=np.int32)
        monomer_idx[mi, : len(idxs)] = idxs
        if len(idxs) < max_atoms:
            monomer_idx[mi, len(idxs) :] = idxs[0] if len(idxs) else 0

    pairs = dimer_permutations(int(n_monomers))
    dimer_idx = np.zeros((len(pairs), max_atoms), dtype=np.int32)
    dimer_n_a = np.zeros(len(pairs), dtype=np.int32)
    dimer_n_b = np.zeros(len(pairs), dtype=np.int32)
    for di, (a, b) in enumerate(pairs):
        ia = np.arange(offsets[a], offsets[a + 1], dtype=np.int32)
        ib = np.arange(offsets[b], offsets[b + 1], dtype=np.int32)
        idxs = np.concatenate([ia, ib])
        dimer_idx[di, : len(idxs)] = idxs
        if len(idxs) < max_atoms:
            dimer_idx[di, len(idxs) :] = idxs[0] if len(idxs) else 0
        dimer_n_a[di] = len(ia)
        dimer_n_b[di] = len(ib)

    return monomer_idx, dimer_idx, dimer_n_a, dimer_n_b, max_atoms


def mic_displacement_numpy(a: np.ndarray, b: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if cell is None:
        return d
    # Cubic or general row-vector cell: use fractional MIC
    cell = np.asarray(cell, dtype=np.float64)
    inv = np.linalg.inv(cell.T)
    frac = (inv @ d.T).T
    frac = frac - np.round(frac)
    return (frac @ cell).reshape(3)


def dimer_com_distance_numpy(
    positions: np.ndarray,
    dimer_indices: np.ndarray,
    n_a: int,
    n_b: int,
    cell: Optional[np.ndarray],
) -> float:
    pos = positions[dimer_indices]
    max_n = pos.shape[0]
    com_a = pos[:n_a].mean(axis=0)
    com_b = pos[n_a : n_a + n_b].mean(axis=0)
    d = mic_displacement_numpy(com_a, com_b, cell)
    return float(np.linalg.norm(d))


def count_near_dimer_pairs(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: int | Sequence[int],
    *,
    mm_switch_on: float = 7.0,
    box_side_A: Optional[float] = None,
    cell: Optional[np.ndarray] = None,
    free_space: bool = False,
) -> dict[str, float | int | bool]:
    """Count dimer pairs with COM distance < ``mm_switch_on`` (same rule as ML sparse path)."""
    pos = np.asarray(positions, dtype=np.float64)
    if cell is None and box_side_A is not None:
        s = float(box_side_A)
        cell = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]], dtype=np.float64)

    _, dimer_idx, dimer_n_a, dimer_n_b, _ = build_monomer_dimer_index_arrays(
        n_monomers, atoms_per_monomer
    )
    n_dimers = len(dimer_idx)
    dists = np.array(
        [
            dimer_com_distance_numpy(pos, dimer_idx[di], int(dimer_n_a[di]), int(dimer_n_b[di]), cell)
            for di in range(n_dimers)
        ],
        dtype=np.float64,
    )
    near = dists < float(mm_switch_on)
    n_near = int(np.sum(near))
    cap = resolve_max_active_dimers(
        n_monomers,
        n_dimers_total=n_dimers,
        free_space=free_space,
    )
    return {
        "n_monomers": int(n_monomers),
        "n_dimers_total": n_dimers,
        "n_near_mm_switch_on": n_near,
        "mm_switch_on_A": float(mm_switch_on),
        "max_active_dimers_cap": cap,
        "free_space": bool(free_space),
        "cap_saturated": n_near > cap,
        "cap_margin": cap - n_near,
        "min_near_dist_A": float(np.min(dists[near])) if n_near else float("nan"),
        "max_near_dist_A": float(np.max(dists[near])) if n_near else float("nan"),
    }


def validate_sparse_dimer_cap(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: int | Sequence[int],
    *,
    mm_switch_on: float = 7.0,
    box_side_A: Optional[float] = None,
    max_active_dimers: Optional[int] = None,
    free_space: bool = False,
) -> dict[str, float | int | bool | str]:
    """Return statistics and a short verdict for the sparse dimer slot cap."""
    stats = count_near_dimer_pairs(
        positions,
        n_monomers,
        atoms_per_monomer,
        mm_switch_on=mm_switch_on,
        box_side_A=box_side_A,
        free_space=free_space,
    )
    cap = (
        resolve_max_active_dimers(
            n_monomers,
            n_dimers_total=int(stats["n_dimers_total"]),
            explicit=max_active_dimers,
            free_space=free_space,
        )
        if max_active_dimers is not None
        else int(stats["max_active_dimers_cap"])
    )
    stats["max_active_dimers_cap"] = cap
    n_near = int(stats["n_near_mm_switch_on"])
    stats["cap_saturated"] = n_near > cap
    stats["cap_margin"] = cap - n_near
    phys_batch = int(n_monomers) + cap

    if stats["cap_saturated"]:
        verdict = (
            f"FAIL: {n_near} near dimers exceed cap {cap} — ML may drop "
            f"{n_near - cap} pairs (raise MMML_MLPOT_MAX_ACTIVE_DIMERS or --ml-max-active-dimers)."
        )
        ok = False
    elif stats["cap_margin"] < 50:
        verdict = (
            f"WARN: only {stats['cap_margin']} slots below {n_near} near dimers — "
            f"consider raising cap before denser configurations."
        )
        ok = True
    else:
        verdict = f"OK: cap {cap} covers {n_near} near dimers (margin {stats['cap_margin']})."
        ok = True

    stats["verdict"] = verdict
    stats["ok"] = ok
    stats["physnet_systems_per_step"] = int(n_monomers) + min(n_near, cap)
    stats["physnet_systems_padded_batch"] = phys_batch
    return stats
