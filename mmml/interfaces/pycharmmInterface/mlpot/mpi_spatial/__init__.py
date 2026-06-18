"""Spatial MPI decomposition for MLpot (halo + sparse dimers).

Design package for Phase 2 multi-rank ML. Single-rank callers can use
``domain`` / ``active_set`` / ``dedup`` without MPI; ``force_exchange`` and
``pool`` add optional mpi4py collectives when available.
"""

from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    RankActiveSet,
    build_all_rank_active_sets,
    build_rank_active_set,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.dedup import (
    assign_canonical_dimer_owner,
    deduplicate_rank_active_sets,
    union_active_dimer_ids,
    verify_unique_dimer_coverage,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    SpatialDomainGrid,
    compute_monomer_coms,
    halo_radius_from_cutoffs,
    monomers_in_extended_domain,
    rank_for_com,
    resolve_halo_radius,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_info import (
    DomdecApiSurvey,
    survey_domdec_api,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange import (
    merge_partial_forces,
    mpi_allreduce_energy,
    mpi_allreduce_forces,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.batch_builder import (
    SpatialBatchIndices,
    build_spatial_batch_indices,
    make_spatial_domain_grid,
    per_rank_physnet_budget,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.pool import (
    MlpotPoolConfig,
    gather_active_system_counts,
)

__all__ = [
    "DomdecApiSurvey",
    "MlpotPoolConfig",
    "RankActiveSet",
    "SpatialBatchIndices",
    "SpatialDomainGrid",
    "assign_canonical_dimer_owner",
    "build_all_rank_active_sets",
    "build_rank_active_set",
    "build_spatial_batch_indices",
    "compute_monomer_coms",
    "deduplicate_rank_active_sets",
    "gather_active_system_counts",
    "halo_radius_from_cutoffs",
    "make_spatial_domain_grid",
    "merge_partial_forces",
    "monomers_in_extended_domain",
    "mpi_allreduce_energy",
    "mpi_allreduce_forces",
    "per_rank_physnet_budget",
    "rank_for_com",
    "resolve_halo_radius",
    "survey_domdec_api",
    "union_active_dimer_ids",
    "verify_unique_dimer_coverage",
]
