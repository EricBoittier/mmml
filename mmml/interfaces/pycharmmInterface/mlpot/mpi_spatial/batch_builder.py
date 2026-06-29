"""Build per-rank PhysNet batch index lists for spatial MPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    RankActiveSet,
    build_rank_active_set,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    SpatialDomainGrid,
    halo_radius_from_cutoffs,
    resolve_halo_radius,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.dedup import (
    canonical_dimer_owner_ranks,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    global_near_dimer_mask,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import compute_monomer_coms


@dataclass(frozen=True)
class SpatialBatchIndices:
    """Owned monomer and active dimer global indices for one MPI rank."""

    rank: int
    owned_monomers: np.ndarray
    active_dimer_indices: np.ndarray
    n_monomers_global: int
    n_dimers_global: int
    physnet_systems: int

    @property
    def n_owned_monomers(self) -> int:
        return int(self.owned_monomers.shape[0])

    @property
    def n_active_dimers(self) -> int:
        return int(self.active_dimer_indices.shape[0])


def make_spatial_domain_grid(
    box_side_A: float,
    n_ranks: int,
    cutoff_params: CutoffParameters,
    *,
    physnet_cutoff: float = 6.0,
) -> SpatialDomainGrid:
    """Construct domain grid from box and cutoff parameters."""
    halo = halo_radius_from_cutoffs(cutoff_params, physnet_cutoff=physnet_cutoff)
    return SpatialDomainGrid(
        box_side_A=float(box_side_A),
        n_ranks=int(n_ranks),
        halo_radius_A=float(halo),
    )


def build_spatial_batch_indices(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
    grid: SpatialDomainGrid,
    rank: int,
    cutoff_params: CutoffParameters,
) -> SpatialBatchIndices:
    """Return owned monomer + active dimer indices for PhysNet on ``rank``."""
    pairs, near = global_near_dimer_mask(
        positions,
        n_monomers,
        atoms_per_monomer,
        mm_switch_on=float(cutoff_params.mm_switch_on),
        box_side_A=grid.box_side_A,
    )
    coms = compute_monomer_coms(positions, n_monomers, atoms_per_monomer)
    cell = np.diag([grid.box_side_A] * 3)
    owners = canonical_dimer_owner_ranks(coms, pairs, near, grid, cell=cell)
    active_set = build_rank_active_set(
        positions,
        n_monomers,
        atoms_per_monomer,
        grid,
        rank,
        mm_switch_on=float(cutoff_params.mm_switch_on),
        dimer_owner_ranks=owners,
        pairs=pairs,
        near=near,
    )
    owned = np.asarray(active_set.owned_monomers, dtype=np.int32)
    dimers = np.asarray(active_set.active_dimer_indices, dtype=np.int32)
    n_dimers = int(n_monomers) * (int(n_monomers) - 1) // 2
    return SpatialBatchIndices(
        rank=int(rank),
        owned_monomers=owned,
        active_dimer_indices=dimers,
        n_monomers_global=int(n_monomers),
        n_dimers_global=n_dimers,
        physnet_systems=int(owned.shape[0] + dimers.shape[0]),
    )


def per_rank_physnet_budget(
    n_monomers: int,
    max_active_dimers: int,
    n_ranks: int,
) -> int:
    """Upper bound on PhysNet systems per rank for cap checks."""
    del n_ranks
    return int(n_monomers) + int(max_active_dimers)
