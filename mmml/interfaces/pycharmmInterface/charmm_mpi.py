"""MPI setup for DOMDEC-linked ``libcharmm.so`` in serial ``python`` runs."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_MPI_LDD_KEYWORDS = (
    "libmpi",
    "libopen-pal",
    "libpmix",
    "libmpi_mpifh",
    "libhwloc",
    "libevent",
)

_pmix_preloaded = False


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
    return bool(os.environ.get("OMPI_COMM_WORLD_SIZE")) and not _under_mpirun()


def _bootstrap_charmm_lib_dir_from_setup() -> None:
    if os.environ.get("CHARMM_LIB_DIR"):
        return
    repo_root = Path(__file__).resolve().parents[3]
    setup_file = repo_root / "CHARMMSETUP"
    if not setup_file.is_file():
        return
    for line in setup_file.read_text(encoding="utf-8").splitlines():
        if "CHARMM_LIB_DIR" in line and "=" in line:
            os.environ.setdefault("CHARMM_LIB_DIR", line.split("=", 1)[1].strip())
        if "CHARMM_HOME" in line and "=" in line:
            os.environ.setdefault("CHARMM_HOME", line.split("=", 1)[1].strip())


def _charmm_lib_path() -> Path | None:
    _bootstrap_charmm_lib_dir_from_setup()
    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip()
    if not lib_dir:
        return None
    for name in ("libcharmm.so", "charmm.so"):
        candidate = Path(lib_dir) / name
        if candidate.is_file():
            return candidate
    return None


def _run_ldd(lib: Path) -> str:
    try:
        proc = subprocess.run(
            ["ldd", str(lib)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _parse_ldd_mpi_library_dirs(ldd_stdout: str) -> tuple[str, ...]:
    dirs: list[str] = []
    seen: set[str] = set()
    for line in ldd_stdout.splitlines():
        if "=>" not in line:
            continue
        low = line.lower()
        if not any(key in low for key in _MPI_LDD_KEYWORDS):
            continue
        try:
            _, _, rest = line.partition("=>")
            libpath = rest.split("(", 1)[0].strip()
        except IndexError:
            continue
        if not libpath.startswith("/"):
            continue
        lib_dir = str(Path(libpath).parent.resolve())
        if lib_dir not in seen:
            seen.add(lib_dir)
            dirs.append(lib_dir)
    return tuple(dirs)


def _parse_ldd_library_paths(ldd_stdout: str, *, keyword: str) -> Path | None:
    for line in ldd_stdout.splitlines():
        if keyword not in line.lower() or "=>" not in line:
            continue
        try:
            _, _, rest = line.partition("=>")
            libpath = rest.split("(", 1)[0].strip()
        except IndexError:
            continue
        if libpath.startswith("/"):
            return Path(libpath)
    return None


@lru_cache(maxsize=1)
def charmm_mpi_library_dirs() -> tuple[str, ...]:
    if _truthy("MMML_NO_CHARMM_MPI"):
        return ()
    lib = _charmm_lib_path()
    if lib is None:
        return ()
    extra = [
        p.strip()
        for p in (os.environ.get("MMML_MPI_LD_PATH_EXTRA") or "").split(os.pathsep)
        if p.strip()
    ]
    dirs = list(_parse_ldd_mpi_library_dirs(_run_ldd(lib)))
    for path in extra:
        if path not in dirs:
            dirs.append(path)
    return tuple(dirs)


@lru_cache(maxsize=1)
def charmm_pmix_library_path() -> Path | None:
    lib = _charmm_lib_path()
    if lib is None:
        return None
    return _parse_ldd_library_paths(_run_ldd(lib), keyword="libpmix")


@lru_cache(maxsize=1)
def charmm_lib_links_mpi() -> bool:
    if _truthy("MMML_CHARMM_MPI"):
        return True
    if _truthy("MMML_NO_CHARMM_MPI"):
        return False
    lib = _charmm_lib_path()
    if lib is None:
        return False
    return "libmpi" in _run_ldd(lib).lower()


def _needs_mpi_setup() -> bool:
    return charmm_lib_links_mpi() or _openmpi_env_without_launch() or _under_mpirun()


def mpi_library_path_export() -> str:
    dirs = charmm_mpi_library_dirs()
    if not dirs:
        return ""
    prefix = os.pathsep.join(dirs)
    return f"export LD_LIBRARY_PATH={prefix}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"


def ensure_charmm_mpi_library_path() -> list[str]:
    if _truthy("MMML_NO_MPI_LD_PATH"):
        return []
    dirs = charmm_mpi_library_dirs()
    if not dirs:
        return []
    cur_parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    prepended: list[str] = []
    for lib_dir in dirs:
        if lib_dir not in cur_parts:
            prepended.append(lib_dir)
    if prepended:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(prepended + cur_parts)
    return prepended


def _preload_pmix_global() -> None:
    global _pmix_preloaded
    if _pmix_preloaded or _truthy("MMML_NO_MPI_LD_PATH"):
        return
    pmix = charmm_pmix_library_path()
    if pmix is None or not pmix.is_file():
        return
    try:
        ctypes.CDLL(str(pmix.resolve()), mode=ctypes.RTLD_GLOBAL)
        _pmix_preloaded = True
    except OSError as exc:
        print(
            f"mmml: failed to preload {pmix} ({exc}). Try:\n  "
            + mpi_library_path_export(),
            file=sys.stderr,
            flush=True,
        )


def prepare_charmm_mpi_runtime() -> None:
    """Apply MPI/PMIx library path and preload before ``import pycharmm``."""
    if not charmm_lib_links_mpi():
        return
    ensure_charmm_mpi_library_path()
    _preload_pmix_global()


def scrub_stale_openmpi_env(*, force: bool = False) -> int:
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
    os.environ.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    os.environ.setdefault("OMPI_MCA_mpi_cuda_support", "0")
    os.environ.setdefault("OMPI_MCA_opal_cuda_support", "0")


def _mpi_comm_valid(*, barrier: bool = False) -> bool:
    if not _mpi4py_available():
        return False
    from mpi4py import MPI

    if not MPI.Is_initialized():
        return False
    try:
        MPI.COMM_WORLD.Get_size()
        if barrier:
            MPI.COMM_WORLD.Barrier()
    except Exception:
        return False
    return True


def _init_mpi_thread_multiple() -> None:
    from mpi4py import MPI

    if MPI.Is_initialized():
        return
    _apply_cuda_mpi_env_defaults()
    try:
        MPI.Init_thread(MPI.THREAD_MULTIPLE)
    except (AttributeError, NotImplementedError):
        MPI.Init()


def _hard_reset_mpi(*, phase: str = "") -> bool:
    """Finalize and re-init MPI (JAX/CUDA can break Fortran MPI in DOMDEC CHARMM)."""
    if not _mpi4py_available():
        return False
    from mpi4py import MPI

    _apply_cuda_mpi_env_defaults()
    if MPI.Is_initialized():
        try:
            MPI.Finalize()
        except Exception:
            pass
    try:
        _init_mpi_thread_multiple()
    except Exception:
        return False
    ok = _mpi_comm_valid(barrier=True)
    if ok and phase and not _truthy("MMML_QUIET_MPI"):
        print(f"mmml: MPI hard reset OK ({phase})", flush=True)
    return ok


def _warn_mpi4py_missing(*, removed: int) -> None:
    print(
        "mmml: OpenMPI-linked CHARMM requires MPI in serial runs. "
        + (f"Removed {removed} stale OpenMPI/PMI env var(s). " if removed else "")
        + "Install mpi4py (`uv sync --extra all`) and run serially, or:\n  "
        + mpirun_launch_hint(),
        file=sys.stderr,
        flush=True,
    )


def _warn_invalid_comm(*, phase: str) -> None:
    print(
        f"mmml: MPI communicator invalid {phase}. Set before launch:\n  "
        + mpi_library_path_export()
        + "\n  export OMPI_MCA_mpi_cuda_support=0\n  "
        + mpirun_launch_hint(),
        file=sys.stderr,
        flush=True,
    )


def ensure_mpi_for_charmm_domdec(*, phase: str = "before PyCHARMM import") -> bool:
    prepare_charmm_mpi_runtime()

    if _truthy("MMML_NO_MPI_INIT"):
        return False
    if not _needs_mpi_setup():
        return False

    if not _under_mpirun():
        scrub_stale_openmpi_env()

    if not _mpi4py_available():
        if charmm_lib_links_mpi() or _openmpi_env_without_launch():
            _warn_mpi4py_missing(removed=0)
        return _under_mpirun()

    if not _mpi_comm_valid():
        _init_mpi_thread_multiple()

    if _mpi_comm_valid():
        return True

    _warn_invalid_comm(phase=phase)
    return _under_mpirun()


def recover_mpi_for_charmm_after_jax(*, phase: str = "after JAX GPU warmup") -> bool:
    """Re-sync MPI after JAX/CUDA before MLpot SD (fixes ``domdec_dr_common`` barriers)."""
    if _truthy("MMML_NO_MPI_INIT") or not _needs_mpi_setup():
        return True
    if _under_mpirun():
        return True
    if not _mpi4py_available():
        _warn_mpi4py_missing(removed=0)
        return False

    # Python-side Get_size() can succeed while Fortran CHARMM barriers still fail.
    if charmm_lib_links_mpi() and not _truthy("MMML_NO_MPI_HARD_RESET"):
        return _hard_reset_mpi(phase=phase)

    if _mpi_comm_valid(barrier=True):
        return True

    if _truthy("MMML_NO_MPI_HARD_RESET"):
        _warn_invalid_comm(phase=phase)
        return False

    return _hard_reset_mpi(phase=phase)


def revalidate_mpi_after_cuda(*, phase: str = "after JAX GPU warmup") -> bool:
    return recover_mpi_for_charmm_after_jax(phase=phase)


def mpirun_launch_hint(argv0: str = "mmml md-system") -> str:
    export_line = mpi_library_path_export()
    lines = ["export OMPI_MCA_mpi_cuda_support=0"]
    if export_line:
        lines.append(export_line)
    lines.append(f"mpirun -np 1 {argv0} ...")
    return "\n".join(lines)


_apply_cuda_mpi_env_defaults()
