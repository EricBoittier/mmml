"""MPI force reduction for spatial ML decomposition."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def merge_partial_forces(
    partials: list[np.ndarray],
    *,
    n_atoms: Optional[int] = None,
) -> np.ndarray:
    """Sum per-rank partial force arrays (reference path for tests, no MPI)."""
    if not partials:
        raise ValueError("partials must be non-empty")
    stacked = [np.asarray(p, dtype=np.float64) for p in partials]
    if n_atoms is None:
        n_atoms = max(s.shape[0] for s in stacked)
    total = np.zeros((int(n_atoms), 3), dtype=np.float64)
    for arr in stacked:
        n = min(arr.shape[0], int(n_atoms))
        total[:n] += arr[:n]
    return total


def mpi_allreduce_forces(
    forces: np.ndarray,
    comm: Any = None,
) -> np.ndarray:
    """Elementwise sum of forces across MPI ranks (mpi4py when available)."""
    from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized

    ensure_charmm_mpi_initialized()
    local = np.asarray(forces, dtype=np.float64)
    if comm is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        except Exception:
            return local.copy()
    if comm.Get_size() <= 1:
        return local.copy()
    from mpi4py import MPI

    out = np.empty_like(local)
    comm.Allreduce([local, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)
    return out


def mpi_allreduce_energy(energy: float, comm: Any = None) -> float:
    """Sum ML energy contributions across MPI ranks."""
    from mmml.interfaces.pycharmmInterface.charmm_mpi import ensure_charmm_mpi_initialized

    ensure_charmm_mpi_initialized()
    local = float(energy)
    if comm is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        except Exception:
            return local
    if comm.Get_size() <= 1:
        return local
    from mpi4py import MPI

    buf = np.array([local], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    comm.Allreduce([buf, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)
    return float(out[0])
