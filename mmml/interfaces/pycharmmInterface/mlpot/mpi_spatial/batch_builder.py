"""Build per-rank PhysNet batch index lists for spatial MPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

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

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import DomdecAlignedGrid


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


def make_domdec_aligned_grid(
    box_side_A: float,
    cutoff_params: CutoffParameters,
    *,
    n_ranks_fallback: int = 1,
    physnet_cutoff: float = 6.0,
) -> "DomdecAlignedGrid":
    """Construct a :class:`DomdecAlignedGrid`; auto-reads NDIR from CHARMM DOMDEC.

    When DOMDEC is active the grid uses ``ndomx`` (Nx from ``domdec ndir``) as
    ``n_ranks``, matching CHARMM's own domain assignment.  When DOMDEC is
    inactive it falls back to ``n_ranks_fallback`` (pass ``mpi_size`` here),
    preserving the existing Tier 2 COM-slab behaviour.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import DomdecAlignedGrid

    halo = halo_radius_from_cutoffs(cutoff_params, physnet_cutoff=physnet_cutoff)
    return DomdecAlignedGrid(
        box_side_A=float(box_side_A),
        halo_radius_A=float(halo),
        n_ranks_fallback=int(n_ranks_fallback),
    )


def build_domdec_spatial_batch_indices(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
    grid: "DomdecAlignedGrid",
    rank: int,
    cutoff_params: CutoffParameters,
) -> SpatialBatchIndices:
    """Return owned monomer + active dimer indices for PhysNet on ``rank``.

    When DOMDEC is active and ctypes symbols are available, ``owned_monomers``
    is read directly from Fortran's ``domdec_common`` / ``domdec_local`` rather
    than computed from COM slabs.  Dimer visibility and ownership continue to
    use COM arithmetic because DOMDEC does not expose dimer assignment.

    When DOMDEC is inactive the call falls through to
    :func:`build_spatial_batch_indices` on the inner :class:`SpatialDomainGrid`,
    preserving Tier 2 COM-slab behaviour exactly.
    """
    inner_grid = grid.grid  # underlying SpatialDomainGrid

    if not grid.domdec_active:
        return build_spatial_batch_indices(
            positions, n_monomers, atoms_per_monomer, inner_grid, rank, cutoff_params
        )

    # DOMDEC active path: derive owned monomers from Fortran atom ownership.
    local_atoms = grid.get_local_atom_indices()
    domdec_owned = grid.molecules_owned_by_this_rank(
        local_atoms, atoms_per_monomer, n_monomers
    )

    # Dimers: COM-based (DOMDEC does not expose dimer assignment).
    pairs, near = global_near_dimer_mask(
        positions,
        n_monomers,
        atoms_per_monomer,
        mm_switch_on=float(cutoff_params.mm_switch_on),
        box_side_A=inner_grid.box_side_A,
    )
    coms = compute_monomer_coms(positions, n_monomers, atoms_per_monomer)
    cell = np.diag([inner_grid.box_side_A] * 3)
    owners = canonical_dimer_owner_ranks(coms, pairs, near, inner_grid, cell=cell)
    active_set = build_rank_active_set(
        positions,
        n_monomers,
        atoms_per_monomer,
        inner_grid,
        rank,
        mm_switch_on=float(cutoff_params.mm_switch_on),
        dimer_owner_ranks=owners,
    )

    owned = np.asarray(domdec_owned, dtype=np.int32)
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
