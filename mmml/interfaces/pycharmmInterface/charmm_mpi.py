"""MPI setup for DOMDEC-linked ``libcharmm.so`` in serial ``python`` runs."""

from __future__ import annotations

import ctypes
import os
import platform
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
_IS_DARWIN = platform.system() == "Darwin"


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
    names = ("libcharmm.so", "libcharmm.dylib", "charmm.so", "charmm.dylib")
    for name in names:
        candidate = Path(lib_dir) / name
        if candidate.is_file():
            return candidate
    lib_subdir = Path(lib_dir) / "lib"
    for name in names:
        candidate = lib_subdir / name
        if candidate.is_file():
            return candidate
    return None


def charmm_lib_available() -> bool:
    """Return True when ``libcharmm.so`` is present under ``CHARMM_LIB_DIR``."""
    return _charmm_lib_path() is not None


def _run_ldd(lib: Path) -> str:
    try:
        if _IS_DARWIN:
            proc = subprocess.run(
                ["otool", "-L", str(lib)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
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


def _linked_library_path(line: str) -> str | None:
    """Extract an absolute shared-library path from ``ldd`` or ``otool -L`` output."""
    line = line.strip()
    if not line or line.endswith(":"):
        return None
    if "=>" in line:
        _, _, rest = line.partition("=>")
        libpath = rest.split("(", 1)[0].strip()
    elif " (compatibility" in line:
        if line.startswith("\t"):
            line = line[1:]
        libpath = line.split(" (compatibility", 1)[0].strip()
    else:
        return None
    if libpath.startswith("/"):
        return libpath
    return None


def _parse_ldd_mpi_library_dirs(ldd_stdout: str) -> tuple[str, ...]:
    dirs: list[str] = []
    seen: set[str] = set()
    for line in ldd_stdout.splitlines():
        low = line.lower()
        if not any(key in low for key in _MPI_LDD_KEYWORDS):
            continue
        libpath = _linked_library_path(line)
        if libpath is None:
            continue
        lib_dir = str(Path(libpath).parent.resolve())
        if lib_dir not in seen:
            seen.add(lib_dir)
            dirs.append(lib_dir)
    return tuple(dirs)


def _parse_ldd_library_paths(ldd_stdout: str, *, keyword: str) -> Path | None:
    for line in ldd_stdout.splitlines():
        if keyword not in line.lower():
            continue
        libpath = _linked_library_path(line)
        if libpath is not None:
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
    var = "DYLD_LIBRARY_PATH" if _IS_DARWIN else "LD_LIBRARY_PATH"
    return f"export {var}={prefix}${{{var}:+:${var}}}"


def _libmpi_paths_from_ldd(ldd_stdout: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for line in ldd_stdout.splitlines():
        low = line.lower()
        if "libmpi" not in low:
            continue
        libpath = _linked_library_path(line)
        if libpath is None:
            continue
        libmpi_path = Path(libpath)
        if libmpi_path in seen:
            continue
        seen.add(libmpi_path)
        paths.append(libmpi_path)
    # Prefer the OpenMPI prefix used to build libcharmm over distro /usr/lib.
    paths.sort(key=lambda p: (str(p).startswith("/usr"), str(p)))
    return paths


def _mpirun_for_libmpi(libmpi: Path) -> Path | None:
    lib_dir = libmpi.parent
    for candidate in (
        lib_dir.parent / "bin" / "mpirun",
        lib_dir / "mpirun",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _openmpi_root_mpirun() -> Path | None:
    """Optional ``OPENMPI_ROOT/bin/mpirun`` (same convention as ``scripts/rebuild_charmm_mlpot.sh``)."""
    root = (os.environ.get("OPENMPI_ROOT") or "").strip()
    if not root:
        return None
    candidate = Path(root).expanduser() / "bin" / "mpirun"
    if candidate.is_file():
        return candidate.resolve()
    return None


@lru_cache(maxsize=1)
def charmm_mpirun_path() -> Path | None:
    """``mpirun`` from the same OpenMPI install as ``libcharmm.so``.

    Resolution order: ``MMML_MPIRUN``, then ``libmpi.so`` from ``ldd libcharmm.so``
    → ``../bin/mpirun``, then ``OPENMPI_ROOT/bin/mpirun`` as fallback.

    ``ldd`` is preferred over ``OPENMPI_ROOT`` so a distro ``OPENMPI_ROOT=/usr`` does
    not mask a custom OpenMPI prefix linked into ``libcharmm.so``.
    When auto-discovery fails, set ``MMML_MPIRUN`` to the launcher that matches
    the ``libmpi`` line from ``ldd`` (do not assume a distro layout).
    """
    override = (os.environ.get("MMML_MPIRUN") or "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    lib = _charmm_lib_path()
    if lib is not None:
        for libmpi in _libmpi_paths_from_ldd(_run_ldd(lib)):
            found = _mpirun_for_libmpi(libmpi)
            if found is not None:
                return found

    from_openmpi_root = _openmpi_root_mpirun()
    if from_openmpi_root is not None:
        return from_openmpi_root

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


def mpi_diagnostic_env_defaults() -> None:
    """OpenMPI MCA defaults for clearer fatal-error output (idempotent)."""
    if _truthy("MMML_NO_MPI_ABORT_STACK"):
        return
    os.environ.setdefault("OMPI_MCA_orte_abort_print_stack", "1")
    # Fewer aggregated PRRTE help blocks (often empty when PRRTE lacks Sphinx docs).
    os.environ.setdefault("OMPI_MCA_orte_base_help_aggregate", "0")


def mpi_mpirun_extra_args() -> list[str]:
    """Extra ``mpirun`` argv tokens for crash diagnostics."""
    if _truthy("MMML_NO_MPI_ABORT_STACK"):
        return []
    args = ["--mca", "orte_abort_print_stack", "1"]
    if _truthy("MMML_MPI_VERBOSE"):
        args.extend(
            [
                "--mca",
                "plm_base_verbose",
                "10",
                "--mca",
                "prte_base_verbose",
                "10",
            ]
        )
    return args


def explain_mpi_crash(exit_code: int, *, argv0: str = "mmml md-system") -> None:
    """Print actionable hints after SIGSEGV/SIGABRT under ``mpirun``."""
    if exit_code not in (134, 139, -6, -11):
        return
    sig = "SIGABRT" if exit_code in (134, -6) else "SIGSEGV"
    lines = [
        f"mmml: MPI job ended with {sig} (exit {exit_code}).",
        "  Ignore PRRTE 'built without Sphinx' help lines — they are launcher noise.",
        "  For source-level backtraces:",
        f"    1. ./scripts/rebuild_charmm_mlpot.sh --debug",
        f"    2. MMML_MPI_GDB=1 ./scripts/mmml-charmm-mpirun.sh {argv0} ...",
        "    3. gdb -batch -ex run -ex 'thread apply all bt' -ex quit --args \\",
        "         <python> -m mmml.cli.__main__ md-system ...",
    ]
    if exit_code in (139, -11):
        lines.append(
            "  MLpot ``upinb`` segfaults: use vendored pycharmm (skip_iblo_inb_update), "
            "OMP_NUM_THREADS=1, and mmml-charmm-mpirun.sh."
        )
        lines.append(
            "  MLpot SD MPI segfaults (``send_coord_to_recip`` / ``PMPI_Free_mem``, or "
            "``ext_bond_update`` in ``gete``): sync mmml (JAX on CPU until after MLpot SD; "
            "default: skip ``domdec off``); rebuild with "
            "./scripts/rebuild_charmm_mlpot.sh --no-domdec."
        )
    print("\n".join(lines), file=sys.stderr, flush=True)


def mpi_shell_setup_lines() -> list[str]:
    """Shell ``export`` lines for LD_LIBRARY_PATH, PATH, and OpenMPI MCA vars."""
    mpi_diagnostic_env_defaults()
    lines = [
        "unset OMPI_MCA_mpi_cuda_support",
        "export OMPI_MCA_opal_cuda_support=0",
    ]
    if not _truthy("MMML_NO_MPI_ABORT_STACK"):
        lines.extend(
            [
                "export OMPI_MCA_orte_abort_print_stack=1",
                "export OMPI_MCA_orte_base_help_aggregate=0",
            ]
        )
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
    var = "DYLD_LIBRARY_PATH" if _IS_DARWIN else "LD_LIBRARY_PATH"
    cur_parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    prepended: list[str] = []
    for lib_dir in dirs:
        if lib_dir not in cur_parts:
            prepended.append(lib_dir)
    if prepended:
        os.environ[var] = os.pathsep.join(prepended + cur_parts)
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
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        sanitize_xla_flags_env,
    )

    sanitize_xla_flags_env(quiet=True)
    prepare_charmm_mpi_runtime()
    if charmm_lib_links_mpi():
        os.environ.setdefault("MMML_NO_JAX_COMPILE_THREADS", "1")
    _pin_charmm_openmp_for_serial_mlpot()
    if _under_mpirun():
        from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
            pin_cuda_for_spatial_mpi,
        )

        pin_cuda_for_spatial_mpi()
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
    mpi_diagnostic_env_defaults()


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
    from mmml.utils.jax_gpu_warmup import sync_jax_gpu_before_charmm

    sync_jax_gpu_before_charmm(phase=phase)
    _pin_charmm_openmp_for_serial_mlpot()
    if _truthy("MMML_NO_MPI_INIT") or not _needs_mpi_setup():
        return True
    if _under_mpirun():
        if _mpi4py_available() and _mpi_comm_valid(barrier=True):
            return True
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


def defer_jax_warmup_until_after_mlpot_sd() -> bool:
    """Defer JAX GPU warmup until after CHARMM MLpot SD on MPI-linked builds.

    JAX/CUDA activity before the first ``gete`` in MLpot SD can corrupt OpenMPI
    registered-memory pools; the next CHARMM energy eval then segfaults in
    ``send_coord_to_recip`` / ``PMPI_Free_mem`` even after ``domdec off``.
    MM pretreat (no JAX) succeeds on the same system.
    """
    if _truthy("MMML_NO_DEFER_JAX_WARMUP"):
        return False
    if _truthy("MMML_DEFER_JAX_WARMUP_UNTIL_AFTER_SD"):
        return True
    return charmm_lib_links_mpi() and _under_mpirun()


def maybe_rerun_mmml_under_mpirun(
    argv: list[str],
    *,
    subcommand: str = "md-system",
) -> int | None:
    """Re-exec ``mmml <subcommand>`` under ``mpirun -np 1`` for MPI-linked CHARMM.

    Serial ``python -m mmml <subcommand>`` can intermittently segfault in Fortran
    ``upinb`` during MLpot registration.  Launching under the same OpenMPI as
    ``libcharmm.so`` initializes MPI before CHARMM/Python start.
    """
    import subprocess
    import sys

    if _truthy("MMML_NO_MPI_RERUN") or _under_mpirun() or not _needs_mpi_setup():
        return None
    if not charmm_lib_links_mpi():
        return None
    mpirun = charmm_mpirun_path()
    if mpirun is None:
        print(
            f"mmml: MPI-linked CHARMM but no matching OpenMPI mpirun found. "
            f"Set OPENMPI_ROOT or MMML_MPIRUN, or use:\n  "
            + mpirun_launch_hint(f"mmml {subcommand}"),
            flush=True,
        )
        return None
    if str(mpirun).startswith("/usr/bin/"):
        print(
            "mmml: warning: using distro OpenMPI launcher "
            f"{mpirun}; if this fails with PMIx errors, set\n"
            "  export OPENMPI_ROOT=/opt/gcc-14.2.0/openmpi-5.0.5/build\n"
            "  export MMML_MPIRUN=$OPENMPI_ROOT/bin/mpirun\n"
            "or run via ./scripts/mmml-charmm-mpirun.sh",
            flush=True,
        )
    prepare_serial_charmm_mpi_env()
    tail = list(argv)
    if not tail or tail[0] != subcommand:
        tail = [subcommand, *tail]
    cmd = [
        str(mpirun),
        "-np",
        "1",
        *mpi_mpirun_extra_args(),
        sys.executable,
        "-m",
        "mmml.cli.__main__",
        *tail,
    ]
    env = os.environ.copy()
    bindir = str(mpirun.parent)
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if bindir not in path_parts:
        env["PATH"] = os.pathsep.join([bindir, *path_parts])
    print(
        f"mmml: MPI-linked CHARMM — re-launching under OpenMPI for {subcommand}:\n  "
        + " ".join(cmd),
        flush=True,
    )
    proc = subprocess.run(cmd, env=env)
    rc = int(proc.returncode)
    explain_mpi_crash(rc, argv0=f"mmml {subcommand}")
    return rc


def maybe_rerun_md_system_under_mpirun(argv: list[str]) -> int | None:
    """Backward-compatible alias for :func:`maybe_rerun_mmml_under_mpirun`."""
    return maybe_rerun_mmml_under_mpirun(argv, subcommand="md-system")


_apply_cuda_mpi_env_defaults()
