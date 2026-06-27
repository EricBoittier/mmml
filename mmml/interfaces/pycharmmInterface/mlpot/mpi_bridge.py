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


def _rank_size_from_launcher_env() -> Tuple[int, int]:
    """Read rank/size from OpenMPI / PMI launcher env (set under ``mpirun``)."""
    rank_raw = (
        os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMIX_RANK")
        or os.environ.get("PMI_RANK")
        or "0"
    )
    size_raw = (
        os.environ.get("OMPI_COMM_WORLD_SIZE")
        or os.environ.get("PMIX_SIZE")
        or os.environ.get("PMI_SIZE")
        or "1"
    )
    return int(rank_raw), max(1, int(size_raw))


def _mpi4py_is_initialized() -> bool:
    try:
        from mpi4py import MPI

        return bool(MPI.Is_initialized())
    except Exception:
        return False


def mpi_rank_size(comm: Any = None) -> Tuple[int, int]:
    """Return ``(rank, size)`` using mpi4py or OpenMPI env vars.

    CHARMM-linked builds call ``MPI_Init`` from Fortran; mpi4py's ``COMM_WORLD``
    can remain a singleton (size 1) even when ``mpirun -np N`` launched N ranks.
    When launcher env reports ``size > 1`` but mpi4py does not, trust the env.

    Avoid touching ``COMM_WORLD`` before ``MPI_Init`` (CHARMM or mpi4py) so
    mixed OpenMPI installs do not fail ``opal_shmem_base_select`` during import.
    """
    if comm is not None:
        return int(comm.Get_rank()), int(comm.Get_size())
    env_rank, env_size = _rank_size_from_launcher_env()
    if not _mpi4py_is_initialized() and env_size > 1:
        return env_rank, env_size
    try:
        from mpi4py import MPI

        c = MPI.COMM_WORLD
        rank, size = int(c.Get_rank()), int(c.Get_size())
        if size <= 1 and env_size > 1:
            return env_rank, env_size
        return rank, size
    except Exception:
        return env_rank, env_size


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
        from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange import (
            mpi_allreduce_energy,
            mpi_allreduce_forces,
        )

        ensure_charmm_mpi_initialized()

        f_out = mpi_allreduce_forces(np.asarray(forces, dtype=np.float64), comm=comm)
        e_out = mpi_allreduce_energy(float(energy_kcal), comm=comm)
        return f_out[: int(n_atoms)], e_out
    if size <= 1:
        if forces is None:
            raise ValueError("forces required on single rank")
        return np.asarray(forces, dtype=np.float64), float(energy_kcal)

    from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized

    ensure_charmm_mpi_initialized()
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
