"""Rank-0 MLpot MPI bridge for ``np>1`` correctness (Phase 1).

When multiple MPI ranks are active, only rank 0 runs the JAX PhysNet callback;
forces and energy are broadcast to other ranks before subtracting into CHARMM
force arrays. This is a correctness stopgap, not a performance path.

Disable with ``MMML_MLPOT_RANK0_BRIDGE=0`` (falls back to every rank running ML).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

_BRIDGE_ENV = "MMML_MLPOT_RANK0_BRIDGE"


def _truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "yes", "true")


def rank0_bridge_enabled() -> bool:
    """True when rank-0-only MLpot execution is enabled (default on)."""
    return _truthy(_BRIDGE_ENV, default=True)


def mpi_rank_size(comm: Any = None) -> Tuple[int, int]:
    """Return ``(rank, size)`` using mpi4py or OpenMPI env vars."""
    if comm is not None:
        return int(comm.Get_rank()), int(comm.Get_size())
    try:
        from mpi4py import MPI

        c = MPI.COMM_WORLD
        return int(c.Get_rank()), int(c.Get_size())
    except Exception:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
        return rank, max(1, size)


def mlpot_runs_on_this_rank(comm: Any = None) -> bool:
    """True when this rank should execute the MLpot JAX callback."""
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
    )

    rank, size = mpi_rank_size(comm)
    if size > 1 and spatial_mpi_enabled():
        return True
    if not rank0_bridge_enabled():
        return True
    if size <= 1:
        return True
    return rank == 0


def broadcast_mlpot_result(
    forces: Optional[np.ndarray],
    energy_kcal: float,
    n_atoms: int,
    *,
    comm: Any = None,
) -> Tuple[np.ndarray, float]:
    """Broadcast ML forces and energy from rank 0 to all ranks (rank-0 bridge only)."""
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
    )

    rank, size = mpi_rank_size(comm)
    if size > 1 and spatial_mpi_enabled():
        if forces is None:
            raise ValueError("forces required in spatial MPI mode")
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange import (
            mpi_allreduce_energy,
            mpi_allreduce_forces,
        )

        f_out = mpi_allreduce_forces(np.asarray(forces, dtype=np.float64), comm=comm)
        e_out = mpi_allreduce_energy(float(energy_kcal), comm=comm)
        return f_out[: int(n_atoms)], e_out
    if size <= 1:
        if forces is None:
            raise ValueError("forces required on single rank")
        return np.asarray(forces, dtype=np.float64), float(energy_kcal)

    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    from mpi4py import MPI

    e_buf = np.array([energy_kcal if rank == 0 else 0.0], dtype=np.float64)
    comm.Bcast([e_buf, MPI.DOUBLE], root=0)
    energy_out = float(e_buf[0])

    f_buf = np.zeros((int(n_atoms), 3), dtype=np.float64)
    if rank == 0:
        if forces is None:
            raise ValueError("forces required on rank 0")
        f_buf[:] = np.asarray(forces, dtype=np.float64)[:n_atoms]
    comm.Bcast([f_buf, MPI.DOUBLE], root=0)
    return f_buf, energy_out
