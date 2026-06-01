"""MPI setup for DOMDEC-linked ``libcharmm.so`` in serial ``python`` runs."""

from __future__ import annotations

import os
import sys


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def _under_mpirun() -> bool:
    return any(
        os.environ.get(k) is not None
        for k in (
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "MPI_LOCALRANKID",
        )
    )


def _openmpi_env_without_launch() -> bool:
    """Module loads often set OMPI_COMM_WORLD_SIZE without a valid communicator."""
    return bool(os.environ.get("OMPI_COMM_WORLD_SIZE")) and not _under_mpirun()


def scrub_stale_openmpi_env() -> int:
    """Remove OpenMPI/PMI vars set by modules when not launched under ``mpirun``."""
    if not _openmpi_env_without_launch():
        return 0
    removed = 0
    for key in list(os.environ):
        if key.startswith(("OMPI_", "PMIX_", "PMI_")) or key in ("I_MPI_HYDRATOR",):
            os.environ.pop(key, None)
            removed += 1
    return removed


def ensure_mpi_for_charmm_domdec() -> None:
    """Initialize MPI once before ``import pycharmm`` when using OpenMPI-linked CHARMM.

    Without this, vacuum ``minimize`` / MLpot SD on some clusters fail with::

        MPI_ERR_COMM: invalid communicator

    Set ``MMML_NO_MPI_INIT=1`` to disable. Prefer ``mpirun -np 1`` if problems persist.
    """
    if _truthy("MMML_NO_MPI_INIT"):
        return
    if not (_openmpi_env_without_launch() or _under_mpirun()):
        return
    try:
        from mpi4py import MPI
    except ImportError:
        if _openmpi_env_without_launch():
            n = scrub_stale_openmpi_env()
            print(
                "mmml: mpi4py not installed; removed "
                f"{n} stale OpenMPI/PMI env var(s). "
                "If CHARMM still fails in MPI_Barrier, run: "
                f"{mpirun_launch_hint()}",
                file=sys.stderr,
                flush=True,
            )
        return
    comm = MPI.COMM_WORLD
    try:
        comm.Get_size()
    except Exception:
        if not MPI.Is_initialized():
            MPI.Init()


def mpirun_launch_hint(argv0: str = "mmml md-system") -> str:
    return f"mpirun -np 1 {argv0} ..."
