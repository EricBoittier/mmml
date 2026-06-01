"""MPI setup for DOMDEC-linked ``libcharmm.so`` in serial ``python`` runs."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


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
            "OMPI_MCA_orte_precondition_transports",
        )
    )


def _mpi4py_available() -> bool:
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


def _openmpi_env_without_launch() -> bool:
    """Module loads often set OMPI_COMM_WORLD_SIZE without a valid communicator."""
    return bool(os.environ.get("OMPI_COMM_WORLD_SIZE")) and not _under_mpirun()


def _charmm_lib_path() -> Path | None:
    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip()
    if not lib_dir:
        return None
    for name in ("libcharmm.so", "charmm.so"):
        candidate = Path(lib_dir) / name
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def charmm_lib_links_mpi() -> bool:
    """True when ``libcharmm.so`` is linked against ``libmpi`` (OpenMPI/MPI builds)."""
    if _truthy("MMML_CHARMM_MPI"):
        return True
    if _truthy("MMML_NO_CHARMM_MPI"):
        return False
    lib = _charmm_lib_path()
    if lib is None:
        return False
    try:
        proc = subprocess.run(
            ["ldd", str(lib)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    return "libmpi" in proc.stdout.lower()


def _needs_mpi_setup() -> bool:
    return charmm_lib_links_mpi() or _openmpi_env_without_launch() or _under_mpirun()


def scrub_stale_openmpi_env(*, force: bool = False) -> int:
    """Remove OpenMPI/PMI vars set by modules when not launched under ``mpirun``."""
    if not force and not _openmpi_env_without_launch():
        if not (charmm_lib_links_mpi() and not _under_mpirun()):
            return 0
    if _under_mpirun():
        return 0
    removed = 0
    for key in list(os.environ):
        if key.startswith(("OMPI_", "PMIX_", "PMI_")) or key in ("I_MPI_HYDRATOR",):
            os.environ.pop(key, None)
            removed += 1
    return removed


def _apply_cuda_mpi_env_defaults() -> None:
    """OpenMPI + CUDA defaults (set before ``MPI_Init``)."""
    os.environ.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")


def _mpi_comm_valid() -> bool:
    if not _mpi4py_available():
        return False
    from mpi4py import MPI

    if not MPI.Is_initialized():
        return False
    try:
        MPI.COMM_WORLD.Get_size()
    except Exception:
        return False
    return True


def _init_mpi_thread_multiple() -> None:
    from mpi4py import MPI

    if MPI.Is_initialized():
        return
    _apply_cuda_mpi_env_defaults()
    try:
        required = MPI.THREAD_MULTIPLE
        MPI.Init_thread(required)
    except (AttributeError, NotImplementedError):
        MPI.Init()


def _warn_mpi4py_missing(*, removed: int) -> None:
    print(
        "mmml: OpenMPI-linked CHARMM requires MPI in serial runs. "
        + (f"Removed {removed} stale OpenMPI/PMI env var(s). " if removed else "")
        + "Install mpi4py (`pip install mpi4py`) or run under "
        + mpirun_launch_hint(),
        file=sys.stderr,
        flush=True,
    )


def _warn_invalid_comm(*, phase: str) -> None:
    print(
        f"mmml: MPI communicator invalid {phase} (common after JAX GPU init on "
        "OpenMPI-linked CHARMM). Run:\n  "
        + mpirun_launch_hint()
        + "\nOr install mpi4py and ensure MPI is initialized before PyCHARMM "
        "(set MMML_CHARMM_MPI=1 to force setup).",
        file=sys.stderr,
        flush=True,
    )


def ensure_mpi_for_charmm_domdec(*, phase: str = "before PyCHARMM import") -> bool:
    """Initialize MPI once before ``import pycharmm`` when using OpenMPI-linked CHARMM.

    Without this, vacuum ``minimize`` / MLpot SD on some clusters fail with::

        MPI_ERR_COMM: invalid communicator

    Large decomposed MLpot runs are especially sensitive: JAX GPU warmup can
    invalidate a communicator that was never initialized properly.

    Set ``MMML_NO_MPI_INIT=1`` to disable. Prefer ``mpirun -np 1`` if problems persist.
    Returns True when ``COMM_WORLD`` appears valid after this call.
    """
    if _truthy("MMML_NO_MPI_INIT"):
        return False
    if not _needs_mpi_setup():
        return False

    if not _under_mpirun():
        scrub_stale_openmpi_env()

    try:
        from mpi4py import MPI
    except ImportError:
        if charmm_lib_links_mpi() or _openmpi_env_without_launch():
            _warn_mpi4py_missing(removed=0)
        return False

    if not MPI.Is_initialized():
        _init_mpi_thread_multiple()

    if _mpi_comm_valid():
        return True

    _warn_invalid_comm(phase=phase)
    return False


def revalidate_mpi_after_cuda(*, phase: str = "after JAX GPU warmup") -> bool:
    """Re-check MPI after CUDA/JAX init; large MLpot clusters hit ``MPI_Barrier`` here.

    When launched under ``mpirun``, Fortran CHARMM owns ``MPI_COMM_WORLD`` even if
    ``mpi4py`` is not installed — do not block those runs based on Python-side checks.
    """
    if _truthy("MMML_NO_MPI_INIT") or not _needs_mpi_setup():
        return True

    # mpirun provides a valid communicator to libcharmm.so; mpi4py is optional.
    if _under_mpirun():
        if not _mpi_comm_valid() and _mpi4py_available():
            _warn_invalid_comm(phase=f"{phase} (under mpirun; proceeding anyway)")
        return True

    if _mpi_comm_valid():
        return True

    scrub_stale_openmpi_env(force=True)
    if not _mpi4py_available():
        _warn_mpi4py_missing(removed=0)
        return False

    from mpi4py import MPI

    if not MPI.Is_initialized():
        _init_mpi_thread_multiple()
    if _mpi_comm_valid():
        return True
    _warn_invalid_comm(phase=phase)
    return False


def mpirun_launch_hint(argv0: str = "mmml md-system") -> str:
    return f"mpirun -np 1 {argv0} ..."
