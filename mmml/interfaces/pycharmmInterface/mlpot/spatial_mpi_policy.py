"""Opt-in spatial MPI ML decomposition policy."""

from __future__ import annotations

import os
from typing import Any


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


def sync_spatial_mpi_env(*, explicit: bool | None = None) -> bool:
    """Set ``MMML_MLPOT_SPATIAL_MPI=1`` when spatial ML is requested."""
    if explicit is True or (explicit is None and spatial_mpi_enabled()):
        os.environ[_SPATIAL_ENV] = "1"
        return True
    return False


def sync_spatial_mpi_env_from_args(args: Any) -> bool:
    """Mirror ``--ml-spatial-mpi`` / YAML ``ml_spatial_mpi`` into the process env."""
    if args is None:
        return False
    return sync_spatial_mpi_env(explicit=bool(getattr(args, "ml_spatial_mpi", False)))


def sync_spatial_mpi_env_from_campaign(
    campaign: dict[str, Any] | None,
    args: Any = None,
) -> bool:
    """Set spatial MPI env from campaign defaults, jobs, and parent CLI."""
    enabled = False
    if args is not None and bool(getattr(args, "ml_spatial_mpi", False)):
        enabled = True
    if campaign:
        defaults = campaign.get("defaults")
        if isinstance(defaults, dict) and defaults.get("ml_spatial_mpi"):
            enabled = True
        runs = campaign.get("runs") or campaign.get("jobs") or {}
        if isinstance(runs, dict):
            for job in runs.values():
                if isinstance(job, dict) and job.get("ml_spatial_mpi"):
                    enabled = True
                    break
    return sync_spatial_mpi_env(explicit=enabled if enabled else None)
