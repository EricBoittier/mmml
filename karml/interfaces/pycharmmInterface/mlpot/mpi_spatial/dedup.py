"""Canonical dimer ownership and deduplication across MPI ranks."""

from __future__ import annotations

from typing import Optional

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import RankActiveSet
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    SpatialDomainGrid,
    dimer_com_mic,
)


def assign_canonical_dimer_owner(
    i: int,
    j: int,
    com_i: np.ndarray,
    com_j: np.ndarray,
    grid: SpatialDomainGrid,
    *,
    cell: Optional[np.ndarray] = None,
) -> int:
    """Owning rank for dimer ``(i, j)``: rank of dimer COM (deterministic slab)."""
    del i, j
    com_ij = dimer_com_mic(com_i, com_j, cell)
    return int(grid.rank_for_com(com_ij))


def canonical_dimer_owner_ranks(
    coms: np.ndarray,
    pairs: np.ndarray,
    near_mask: np.ndarray,
    grid: SpatialDomainGrid,
    *,
    cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-dimer owner rank for all near pairs; -1 for non-near slots."""
    n_dimers = pairs.shape[0]
    owners = np.full(n_dimers, -1, dtype=np.int32)
    for di in range(n_dimers):
        if not near_mask[di]:
            continue
        i, j = int(pairs[di, 0]), int(pairs[di, 1])
        owners[di] = assign_canonical_dimer_owner(
            i, j, coms[i], coms[j], grid, cell=cell
        )
    return owners


def deduplicate_rank_active_sets(
    active_sets: list[RankActiveSet],
    dimer_owner_ranks: np.ndarray,
) -> list[RankActiveSet]:
    """Keep only dimers owned by each rank."""
    out: list[RankActiveSet] = []
    for ras in active_sets:
        mask = dimer_owner_ranks[ras.candidate_dimer_indices] == int(ras.rank)
        active = ras.candidate_dimer_indices[mask]
        out.append(
            RankActiveSet(
                rank=ras.rank,
                owned_monomers=ras.owned_monomers,
                ghost_monomers=ras.ghost_monomers,
                monomers_for_ml=ras.monomers_for_ml,
                candidate_dimer_indices=ras.candidate_dimer_indices,
                active_dimer_indices=active,
                dimer_owner_ranks=dimer_owner_ranks[ras.candidate_dimer_indices],
            )
        )
    return out


def union_active_dimer_ids(active_sets: list[RankActiveSet]) -> np.ndarray:
    """Sorted union of active dimer indices across ranks."""
    if not active_sets:
        return np.zeros(0, dtype=np.int32)
    parts = [s.active_dimer_indices for s in active_sets if s.active_dimer_indices.size]
    if not parts:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(parts)).astype(np.int32)


def verify_unique_dimer_coverage(
    active_sets: list[RankActiveSet],
    global_near_dimer_ids: np.ndarray,
) -> bool:
    """True if union of per-rank active dimers equals the global near set."""
    union = union_active_dimer_ids(active_sets)
    global_near = np.unique(np.asarray(global_near_dimer_ids, dtype=np.int32))
    return union.shape == global_near.shape and np.all(union == global_near)
