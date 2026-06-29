"""Per-rank active monomer and dimer sets for spatial ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import dimer_permutations
from mmml.interfaces.pycharmmInterface.cutoffs import DEFAULT_MM_SWITCH_ON
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    mic_displacement_numpy,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    SpatialDomainGrid,
    compute_monomer_coms,
    dimer_com_mic,
)


@dataclass
class RankActiveSet:
    """Monomers and dimers selected for ML evaluation on one MPI rank."""

    rank: int
    owned_monomers: np.ndarray
    ghost_monomers: np.ndarray
    monomers_for_ml: np.ndarray
    candidate_dimer_indices: np.ndarray
    active_dimer_indices: np.ndarray
    dimer_owner_ranks: np.ndarray

    @property
    def n_active_dimers(self) -> int:
        return int(self.active_dimer_indices.shape[0])

    @property
    def n_owned_monomers(self) -> int:
        return int(self.owned_monomers.shape[0])


def _cell_from_box(box_side_A: Optional[float]) -> Optional[np.ndarray]:
    if box_side_A is None:
        return None
    s = float(box_side_A)
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]], dtype=np.float64)


def monomer_pair_com_distances_mic(
    coms: np.ndarray,
    pairs: np.ndarray,
    cell: Optional[np.ndarray],
) -> np.ndarray:
    """MIC distances between monomer COM pairs, shape ``(n_pairs,)``."""
    coms = np.asarray(coms, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int32)
    a = coms[pairs[:, 0]]
    b = coms[pairs[:, 1]]
    d = b - a
    if cell is not None:
        cell = np.asarray(cell, dtype=np.float64)
        inv = np.linalg.inv(cell.T)
        frac = (inv @ d.T).T
        frac = frac - np.round(frac)
        d = frac @ cell
    return np.linalg.norm(d, axis=1)


def global_near_dimer_mask(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
    *,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    box_side_A: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dimer_pair_index array shape (n_dimers,2), near_mask bool)."""
    pairs = np.array(dimer_permutations(int(n_monomers)), dtype=np.int32)
    coms = compute_monomer_coms(positions, n_monomers, atoms_per_monomer)
    cell = _cell_from_box(box_side_A)
    dists = monomer_pair_com_distances_mic(coms, pairs, cell)
    return pairs, dists < float(mm_switch_on)


def build_rank_active_set(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
    grid: SpatialDomainGrid,
    rank: int,
    *,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    dimer_owner_ranks: Optional[np.ndarray] = None,
    pairs: Optional[np.ndarray] = None,
    near: Optional[np.ndarray] = None,
) -> RankActiveSet:
    """Build monomer/dimer active lists for one rank (before deduplication)."""
    coms = compute_monomer_coms(positions, n_monomers, atoms_per_monomer)
    owned = grid.owned_monomer_mask(coms, rank)
    extended = grid.monomers_in_extended_domain(coms, rank)
    ghost = extended & ~owned

    if pairs is None or near is None:
        pairs, near = global_near_dimer_mask(
            positions,
            n_monomers,
            atoms_per_monomer,
            mm_switch_on=mm_switch_on,
            box_side_A=grid.box_side_A,
        )
    cell = _cell_from_box(grid.box_side_A)

    candidate: list[int] = []
    owners: list[int] = []
    for di, ((i, j), is_near) in enumerate(zip(pairs, near)):
        if not is_near:
            continue
        com_ij = dimer_com_mic(coms[i], coms[j], cell)
        visible = extended[i] or extended[j] or grid.monomers_in_extended_domain(
            com_ij.reshape(1, 3), rank
        )[0]
        if not visible:
            continue
        candidate.append(di)
        if dimer_owner_ranks is not None:
            owners.append(int(dimer_owner_ranks[di]))
        else:
            owners.append(grid.rank_for_com(com_ij))

    candidate_arr = np.asarray(candidate, dtype=np.int32)
    owner_arr = np.asarray(owners, dtype=np.int32)
    active = candidate_arr[owner_arr == int(rank)]

    owned_idx = np.nonzero(owned)[0].astype(np.int32)
    ghost_idx = np.nonzero(ghost)[0].astype(np.int32)
    ml_idx = np.unique(np.concatenate([owned_idx, ghost_idx])).astype(np.int32)

    return RankActiveSet(
        rank=int(rank),
        owned_monomers=owned_idx,
        ghost_monomers=ghost_idx,
        monomers_for_ml=ml_idx,
        candidate_dimer_indices=candidate_arr,
        active_dimer_indices=active,
        dimer_owner_ranks=owner_arr,
    )


def build_all_rank_active_sets(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
    grid: SpatialDomainGrid,
    *,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
) -> list[RankActiveSet]:
    """Build per-rank active sets with canonical dimer ownership."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.dedup import (
        canonical_dimer_owner_ranks,
        deduplicate_rank_active_sets,
    )

    coms = compute_monomer_coms(positions, n_monomers, atoms_per_monomer)
    pairs, near = global_near_dimer_mask(
        positions,
        n_monomers,
        atoms_per_monomer,
        mm_switch_on=mm_switch_on,
        box_side_A=grid.box_side_A,
    )
    cell = _cell_from_box(grid.box_side_A)
    owners = canonical_dimer_owner_ranks(coms, pairs, near, grid, cell=cell)

    raw = [
        build_rank_active_set(
            positions,
            n_monomers,
            atoms_per_monomer,
            grid,
            r,
            mm_switch_on=mm_switch_on,
            dimer_owner_ranks=owners,
        )
        for r in range(int(grid.n_ranks))
    ]
    return deduplicate_rank_active_sets(raw, owners)
