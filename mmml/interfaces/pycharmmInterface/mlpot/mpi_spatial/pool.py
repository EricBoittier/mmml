"""Optional ML pool gather/scatter (Phase 2 extension)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import RankActiveSet


@dataclass(frozen=True)
class MlpotPoolConfig:
    """Configuration for a dedicated ML execution pool rank."""

    pool_rank: int = 0
    enabled: bool = False


def gather_active_system_counts(
    active_sets: list[RankActiveSet],
    comm: Any = None,
) -> tuple[int, list[int]]:
    """Return total PhysNet systems (monomers + dimers) and per-rank counts.

  When ``comm`` is None and mpi4py is unavailable, sums locally (single rank).
    """
    per_rank = [int(s.n_owned_monomers) + int(s.n_active_dimers) for s in active_sets]
    total = int(sum(per_rank))
    if comm is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        except Exception:
            return total, per_rank
    if comm.Get_size() <= 1:
        return total, per_rank
    counts = np.array(per_rank, dtype=np.int32)
    total_arr = np.zeros(1, dtype=np.int32)
    from mpi4py import MPI

    comm.Allreduce([counts, MPI.INT], [total_arr, MPI.INT], op=MPI.SUM)
    # total systems is sum of per-rank owned monomers + active dimers (deduped)
    return int(np.sum(counts)), per_rank


def pool_rank_executes_ml(config: MlpotPoolConfig, rank: int) -> bool:
    """True when this rank should run the batched PhysNet forward pass."""
    if not config.enabled:
        return True
    return int(rank) == int(config.pool_rank)
