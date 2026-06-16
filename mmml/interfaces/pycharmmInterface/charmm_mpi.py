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


@lru_cache(maxsize=1)
def charmm_mpirun_path() -> Path | None:
    """``mpirun`` from the same OpenMPI install as ``libcharmm.so`` (not system OpenMPI 3)."""
    override = (os.environ.get("MMML_MPIRUN") or "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    lib = _charmm_lib_path()
    if lib is None:
        return None

    for line in _run_ldd(lib).splitlines():
        low = line.lower()
        if "libmpi.so" not in low or "=>" not in line:
            continue
        try:
            _, _, rest = line.partition("=>")
            libpath = rest.split("(", 1)[0].strip()
        except IndexError:
            continue
        if not libpath.startswith("/"):
            continue
        lib_dir = Path(libpath).parent.resolve()
        for candidate in (
            lib_dir.parent / "bin" / "mpirun",
            lib_dir / "mpirun",
        ):
            if candidate.is_file():
                return candidate.resolve()
    return None


def mpi_path_export() -> str:
    mpirun = charmm_mpirun_path()
    if mpirun is None:
        return ""
    bindir = str(mpirun.parent)
    return f"export PATH={bindir}${{PATH:+:$PATH}}"


def _scrub_deprecated_openmpi_mca_env() -> None:
    """OpenMPI 5+: ``mpi_cuda_support`` is deprecated in favor of ``opal_cuda_support``."""
    os.environ.pop("OMPI_MCA_mpi_cuda_support", None)


def mpi_shell_setup_lines() -> list[str]:
    """Shell ``export`` lines for LD_LIBRARY_PATH, PATH, and OpenMPI MCA vars."""
    lines = [
        "unset OMPI_MCA_mpi_cuda_support",
        "export OMPI_MCA_opal_cuda_support=0",
    ]
    ld = mpi_library_path_export()
    if ld:
        lines.append(ld)
    path_line = mpi_path_export()
    if path_line:
        lines.append(path_line)
    mpirun = charmm_mpirun_path()
    if mpirun is not None:
        lines.append(f"export MMML_MPIRUN={mpirun}")
    return lines


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
    _apply_cuda_mpi_env_defaults()
    _scrub_deprecated_openmpi_mca_env()
    if not charmm_lib_links_mpi():
        return
    ensure_charmm_mpi_library_path()
    _preload_pmix_global()


def _pin_charmm_openmp_for_serial_mlpot() -> None:
    """Serial MLpot + MPI-linked ``libcharmm.so``: OpenMP in ``upinb`` must stay single-threaded.

    Module stacks often export ``OMP_NUM_THREADS`` > 1; combined with JAX/CUDA that
    intermittently segfaults in ``__nbexcl_MOD_upinb``.  Override with
    ``MMML_CHARMM_OMP_THREADS`` or disable via ``MMML_NO_CHARMM_OMP_PIN=1``.
    """
    if _truthy("MMML_NO_CHARMM_OMP_PIN") or not charmm_lib_links_mpi():
        return
    threads = (os.environ.get("MMML_CHARMM_OMP_THREADS") or "1").strip() or "1"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def prepare_serial_charmm_mpi_env() -> None:
    """Env/LD setup only — do **not** call ``MPI_Init`` (CHARMM owns that)."""
    prepare_charmm_mpi_runtime()
    _pin_charmm_openmp_for_serial_mlpot()
    if not _under_mpirun():
        scrub_stale_openmpi_env()


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
    os.environ.setdefault("OMPI_MCA_opal_cuda_support", "0")
    _scrub_deprecated_openmpi_mca_env()


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


def _python_should_own_mpi_init() -> bool:
    """True when mpi4py may call ``MPI_Init`` (never for serial DOMDEC CHARMM by default)."""
    if _under_mpirun():
        return False
    if charmm_lib_links_mpi() and not _truthy("MMML_MPI_PY_INIT"):
        return False
    return True


def _warn_mpi4py_missing(*, removed: int) -> None:
    print(
        "mmml: OpenMPI-linked CHARMM works best under mpirun. "
        + (f"Removed {removed} stale OpenMPI/PMI env var(s). " if removed else "")
        + "For large MLpot clusters use:\n  "
        + mpirun_launch_hint(),
        file=sys.stderr,
        flush=True,
    )


def _warn_invalid_comm(*, phase: str) -> None:
    print(
        f"mmml: MPI communicator check failed {phase}. Launch with:\n  "
        + mpi_library_path_export()
        + "\n  export OMPI_MCA_opal_cuda_support=0\n  "
        + mpirun_launch_hint(),
        file=sys.stderr,
        flush=True,
    )


def ensure_mpi_for_charmm_domdec(*, phase: str = "before PyCHARMM import") -> bool:
    """Prepare MPI env; optionally validate after CHARMM has initialized MPI."""
    prepare_serial_charmm_mpi_env()

    if _truthy("MMML_NO_MPI_INIT") or not _needs_mpi_setup():
        return True

    if _under_mpirun():
        return True

    # Serial DOMDEC CHARMM: libcharmm.so calls MPI_Init — do not init from Python.
    if charmm_lib_links_mpi() and not _python_should_own_mpi_init():
        return True

    if not _mpi4py_available():
        _warn_mpi4py_missing(removed=0)
        return True

    if not _mpi_comm_valid() and _python_should_own_mpi_init():
        _init_mpi_thread_multiple()

    if _mpi_comm_valid():
        return True

    if _mpi4py_available():
        from mpi4py import MPI

        if MPI.Is_initialized():
            return True

    _warn_invalid_comm(phase=phase)
    return False


def recover_mpi_for_charmm_after_jax(*, phase: str = "after JAX warmup") -> bool:
    """Best-effort MPI sync after JAX — never ``MPI_Finalize`` while CHARMM is loaded."""
    if _truthy("MMML_NO_MPI_INIT") or not _needs_mpi_setup():
        return True
    if _under_mpirun():
        return True

    if _mpi4py_available() and _mpi_comm_valid(barrier=True):
        return True

    # CHARMM Fortran MPI may still be valid even when mpi4py is absent or out of sync.
    return True


def revalidate_mpi_after_cuda(*, phase: str = "after JAX GPU warmup") -> bool:
    return recover_mpi_for_charmm_after_jax(phase=phase)


def mpirun_launch_hint(argv0: str = "mmml md-system") -> str:
    lines = mpi_shell_setup_lines()
    mpirun = charmm_mpirun_path()
    runner = str(mpirun) if mpirun is not None else "mpirun"
    lines.append(f"{runner} -np 1 {argv0} ...")
    lines.append("# or: ./scripts/mmml-charmm-mpirun.sh md-system ...")
    return "\n".join(lines)


_apply_cuda_mpi_env_defaults()
