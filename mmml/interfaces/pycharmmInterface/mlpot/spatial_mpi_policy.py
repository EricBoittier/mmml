"""Opt-in spatial MPI ML decomposition policy."""

from __future__ import annotations

import os


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "yes", "true")


_SPATIAL_ENV = "MMML_MLPOT_SPATIAL_MPI"


def spatial_mpi_enabled(explicit: bool | None = None) -> bool:
    """True when per-rank spatial ML decomposition is requested."""
    if explicit is not None:
        return bool(explicit)
    return _truthy(_SPATIAL_ENV, default=False)


def pin_cuda_for_spatial_mpi() -> bool:
    """Pin ``CUDA_VISIBLE_DEVICES`` to local MPI rank when spatial ML is on."""
    if not spatial_mpi_enabled():
        return False
    if not _truthy("MMML_MPI_PIN_GPU_PER_RANK", default=True):
        return False
    local = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    if local is None:
        local = os.environ.get("MPI_LOCALRANKID")
    if local is None:
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(local))
    return True
