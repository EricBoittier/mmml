"""MPI setup for DOMDEC-linked ``libcharmm.so`` in serial ``python`` runs."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Callable

_MPI_LDD_KEYWORDS = (
    "libmpi",
    "libopen-pal",
    "libpmix",
    "libmpi_mpifh",
    "libmpi_usempif08",
    "libmpi_usempi_ignore_tkr",
    "libhwloc",
    "libevent",
)

_pmix_preloaded = False
_opal_preloaded = False
_mpi_libs_preloaded = False
_charmm_mpi_bootstrapped = False
_mpi4py_charmm_configured = False
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
    """True when the ``mpi4py`` package is installed (does not import ``mpi4py.MPI``)."""
    import importlib.util

    try:
        return importlib.util.find_spec("mpi4py") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _openmpi_env_without_launch() -> bool:
    return bool(os.environ.get("OMPI_COMM_WORLD_SIZE")) and not _under_mpirun()


def _charmm_lib_path() -> Path | None:
    from mmml.interfaces.pycharmmInterface.charmm_paths import bootstrap_charmm_env

    bootstrap_charmm_env()
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


def _openmpi_library_dirs_from_prefix(prefix: Path) -> tuple[str, ...]:
    dirs: list[str] = []
    for sub in ("lib", "lib64"):
        candidate = prefix / sub
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved not in dirs:
                dirs.append(resolved)
    return tuple(dirs)


def _fallback_openmpi_library_dirs() -> tuple[str, ...]:
    """Lib dirs for the OpenMPI install matched to ``libcharmm.so`` / ``mpirun``."""
    dirs: list[str] = []
    prefix = openmpi_install_prefix()
    if prefix is not None:
        for path in _openmpi_library_dirs_from_prefix(prefix):
            if path not in dirs:
                dirs.append(path)
    for env_key in ("OPENMPI_ROOT", "EBROOTOPENMPI"):
        root = (os.environ.get(env_key) or "").strip()
        if not root:
            continue
        for path in _openmpi_library_dirs_from_prefix(Path(root).expanduser()):
            if path not in dirs:
                dirs.append(path)
    return tuple(dirs)


@lru_cache(maxsize=1)
def charmm_mpi_library_dirs() -> tuple[str, ...]:
    if _truthy("MMML_NO_CHARMM_MPI"):
        return ()
    lib = _charmm_lib_path()
    dirs: list[str] = []
    if lib is not None:
        dirs = list(_parse_ldd_mpi_library_dirs(_run_ldd(lib)))
    # ldd may show ``libmpi.so.40 => not found`` when OpenMPI modules are unloaded;
    # still prepend the matched mpirun prefix so libcharmm + libmpi_usempif08 resolve.
    for path in _fallback_openmpi_library_dirs():
        if path not in dirs:
            dirs.insert(0, path)
    extra = [
        p.strip()
        for p in (os.environ.get("MMML_MPI_LD_PATH_EXTRA") or "").split(os.pathsep)
        if p.strip()
    ]
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

    # Debian/Ubuntu multiarch: libmpi under /usr/lib/x86_64-linux-gnu, mpirun in /usr/bin.
    if charmm_lib_links_mpi():
        on_path = shutil.which("mpirun")
        if on_path:
            return Path(on_path).resolve()

    return None


def mpi_path_export() -> str:
    mpirun = charmm_mpirun_path()
    if mpirun is None:
        return ""
    bindir = str(mpirun.parent)
    return f"export PATH={bindir}${{PATH:+:$PATH}}"


@lru_cache(maxsize=1)
def openmpi_install_prefix() -> Path | None:
    """OpenMPI install root for the ``mpirun`` matched to ``libcharmm.so``."""
    mpirun = charmm_mpirun_path()
    if mpirun is None:
        return None
    prefix = mpirun.parent.parent
    if (prefix / "lib").is_dir() or (prefix / "lib64").is_dir():
        return prefix
    return None


def openmpi_opal_library_path() -> Path | None:
    """Resolved ``libopen-pal`` from the matched OpenMPI prefix."""
    prefix = openmpi_install_prefix()
    if prefix is None:
        return None
    for sub in ("lib", "lib64"):
        lib_dir = prefix / sub
        if not lib_dir.is_dir():
            continue
        for pattern in ("libopen-pal.so", "libopen-pal.so*"):
            for shared in sorted(lib_dir.glob(pattern)):
                if shared.is_file():
                    return shared.resolve()
    return None


def _dir_has_mca_plugins(directory: Path) -> bool:
    """True when ``directory`` contains at least one ``mca_*.so`` DSO plugin."""
    return any(directory.glob("mca_*.so"))


def openmpi_mca_component_dir() -> Path | None:
    """Directory with OpenMPI MCA DSO plugins (``mca_*.so``) under the matched prefix.

    Incomplete in-tree ``build/`` trees often have ``lib/openmpi/`` with only
    ``libompi_dbg_msgq.so`` — not MCA plugins. Pointing ``mca_base_component_path``
    there makes OpenMPI scan non-MCA libraries and fail (e.g. ``bad prefix``).
    Static builds with no ``mca_*.so`` should leave this unset and rely on
    explicit ``--mca shmem mmap`` plus ``LD_PRELOAD`` of matched ``libopen-pal``.
    """
    override = (os.environ.get("MMML_MPI_MCA_COMPONENT_DIR") or "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_dir() and _dir_has_mca_plugins(path):
            return path
        return None
    prefix = openmpi_install_prefix()
    if prefix is None:
        return None
    for sub in ("lib/openmpi", "lib64/openmpi", "lib", "lib64"):
        candidate = prefix / sub
        if candidate.is_dir() and _dir_has_mca_plugins(candidate):
            return candidate
    return None


def openmpi_shmem_mca_component() -> str | None:
    """Shmem MCA component with a ``mca_shmem_<name>.so`` plugin in the prefix."""
    override = (os.environ.get("MMML_MCA_SHMEM") or "").strip()
    if override and not override.startswith("^"):
        return override
    prefix = openmpi_install_prefix()
    if prefix is None:
        return None
    search: list[Path] = []
    mca_dir = openmpi_mca_component_dir()
    if mca_dir is not None:
        search.append(mca_dir)
    for sub in ("lib/openmpi", "lib64/openmpi", "lib", "lib64"):
        candidate = prefix / sub
        if candidate.is_dir() and candidate not in search:
            search.append(candidate)
    for name in ("mmap", "sysv", "posix"):
        for directory in search:
            if (directory / f"mca_shmem_{name}.so").is_file():
                return name
    return None


def openmpi_shmem_mca_for_launch() -> str | None:
    """Shmem MCA token for ``mpirun`` / env (DSO name or static-built fallback)."""
    override = (os.environ.get("MMML_MCA_SHMEM") or "").strip()
    if override:
        return override
    found = openmpi_shmem_mca_component()
    if found is not None:
        return found
    if openmpi_install_prefix() is None or _truthy("MMML_NO_MPI_SHMEM_FALLBACK"):
        return None
    # In-tree / --disable-dlopen builds often have no mca_shmem_*.so; use built-in mmap.
    fallback = (os.environ.get("MMML_MCA_SHMEM_STATIC_FALLBACK") or "mmap").strip()
    return fallback or None


def _export_openmpi_opal_ld_preload() -> None:
    """``LD_PRELOAD`` matched ``libopen-pal`` for ``orted`` / rank children."""
    if _truthy("MMML_NO_MPI_OPAL_PRELOAD") or _truthy("MMML_NO_MPI_LD_PATH"):
        return
    opal = openmpi_opal_library_path()
    if opal is None:
        return
    path = str(opal)
    var = "DYLD_INSERT_LIBRARIES" if _IS_DARWIN else "LD_PRELOAD"
    parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    if path not in parts:
        os.environ[var] = os.pathsep.join([path, *parts])


def _mpi_preload_sort_key(path: Path) -> tuple[int, str]:
    """Load core ``libmpi`` and Fortran stubs before dependent ``libmpi_*`` shims."""
    name = path.name
    if name.startswith("libmpi.so"):
        return (0, name)
    if "usempi_ignore_tkr" in name:
        return (1, name)
    if "mpifh" in name:
        return (2, name)
    if "usempif08" in name:
        return (3, name)
    return (4, name)


def _openmpi_lib_search_dirs() -> tuple[str, ...]:
    dirs: list[str] = []
    seen: set[str] = set()
    for lib_dir in (*_fallback_openmpi_library_dirs(), *charmm_mpi_library_dirs()):
        if lib_dir not in seen:
            seen.add(lib_dir)
            dirs.append(lib_dir)
    return tuple(dirs)


def _resolve_mpi_library_in_dir(lib_dir: Path, basename: str) -> Path | None:
    direct = lib_dir / basename
    if direct.is_file():
        return direct.resolve()
    stem = basename.split(".so", 1)[0]
    hits = sorted(lib_dir.glob(f"{stem}.so*"))
    return hits[0].resolve() if hits else None


def _unresolved_ldd_library_names(lib: Path) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for line in _run_ldd(lib).splitlines():
        if "=> not found" not in line:
            continue
        left = line.split("=>", 1)[0].strip()
        if not left:
            continue
        name = left.split()[0]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def _openmpi_mpi_library_candidates() -> tuple[Path, ...]:
    """All OpenMPI ``libmpi*.so*`` shims needed before ``libcharmm.so`` loads."""
    found: list[Path] = []
    seen: set[Path] = set()
    lib_dirs = _openmpi_lib_search_dirs()

    lib = _charmm_lib_path()
    if lib is not None:
        for basename in _unresolved_ldd_library_names(lib):
            if "libmpi" not in basename.lower():
                continue
            for lib_dir in lib_dirs:
                resolved = _resolve_mpi_library_in_dir(Path(lib_dir), basename)
                if resolved is None or resolved in seen:
                    continue
                seen.add(resolved)
                found.append(resolved)

    for lib_dir in lib_dirs:
        base = Path(lib_dir)
        if not base.is_dir():
            continue
        for path in sorted(base.glob("libmpi*.so*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)

    found.sort(key=_mpi_preload_sort_key)
    return tuple(found)


def openmpi_mpi_library_paths_for_preload() -> tuple[Path, ...]:
    """Sorted ``libmpi*.so*`` paths for ``LD_PRELOAD`` / ``ctypes`` preload."""
    return _openmpi_mpi_library_candidates()


def _export_openmpi_mpi_ld_preload() -> None:
    """``LD_PRELOAD`` all matched OpenMPI MPI shims (Fortran + C) for rank children."""
    if _truthy("MMML_NO_MPI_LD_PATH") or _truthy("MMML_NO_MPI_MPI_PRELOAD"):
        return
    paths = openmpi_mpi_library_paths_for_preload()
    if not paths:
        return
    var = "DYLD_INSERT_LIBRARIES" if _IS_DARWIN else "LD_PRELOAD"
    parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    prepend = [str(p) for p in paths if str(p) not in parts]
    if prepend:
        os.environ[var] = os.pathsep.join([*prepend, *parts])


def mpi_openmpi_install_env_defaults() -> None:
    """Block distro MCA plugins; prefer CHARMM-linked OpenMPI shmem/pmix.

    On shared GPU nodes, distro OpenMPI 3 plugins under ``/usr/lib/.../openmpi3``
    can load against the wrong ``libopen-pal`` and fail ``opal_shmem_base_select``
    or ``pmix_value_load`` when ``mpirun`` is OpenMPI 5 from ``/opt/gcc-...``.

    Point MCA search at the matched prefix plugin dir and pick an existing shmem
    component (``mmap``/``sysv``/``posix``). Do not set ``OPAL_PREFIX`` for
    incomplete in-tree ``build/`` trees missing ``share/openmpi``.
    """
    if _truthy("MMML_NO_MPI_MCA_PREFIX"):
        return
    os.environ.setdefault("OMPI_MCA_pmix", "^ext3x")
    mca_dir = openmpi_mca_component_dir()
    if mca_dir is not None:
        path = str(mca_dir)
        os.environ.setdefault("OMPI_MCA_component_path", path)
        os.environ.setdefault("OMPI_MCA_mca_base_component_path", path)
    shmem = openmpi_shmem_mca_for_launch()
    if shmem is not None:
        os.environ.setdefault("OMPI_MCA_shmem", shmem)
    _export_openmpi_mpi_ld_preload()
    _export_openmpi_opal_ld_preload()
    override = (os.environ.get("MMML_OPAL_PREFIX") or "").strip()
    if override:
        os.environ.setdefault("OPAL_PREFIX", override)
        return
    prefix = openmpi_install_prefix()
    if prefix is not None and (prefix / "share" / "openmpi").is_dir():
        os.environ.setdefault("OPAL_PREFIX", str(prefix))


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
    args: list[str] = []
    if not _truthy("MMML_NO_MPI_MCA_PREFIX"):
        args.extend(["--mca", "pmix", "^ext3x"])
        mca_dir = openmpi_mca_component_dir()
        if mca_dir is not None:
            args.extend(["--mca", "mca_base_component_path", str(mca_dir)])
        shmem = openmpi_shmem_mca_for_launch()
        if shmem is not None:
            args.extend(["--mca", "shmem", shmem])
    if not _truthy("MMML_NO_MPI_LD_PATH"):
        for var in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "OPENMPI_ROOT",
            "EBROOTOPENMPI",
            "CHARMM_LIB_DIR",
        ):
            if (os.environ.get(var) or "").strip():
                args.extend(["-x", var])
    if _truthy("MMML_NO_MPI_ABORT_STACK"):
        return args
    args.extend(["--mca", "orte_abort_print_stack", "1"])
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


def charmm_libmpi_path() -> Path | None:
    """``libmpi`` shared library linked from ``libcharmm.so`` (``ldd``)."""
    lib = _charmm_lib_path()
    if lib is None:
        return None
    paths = _libmpi_paths_from_ldd(_run_ldd(lib))
    return paths[0] if paths else None


def mpi4py_mpi_extension_path() -> Path | None:
    """Path to the compiled ``mpi4py.MPI`` extension module (no ``MPI_Init``)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("mpi4py.MPI")
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError, RuntimeError):
        return None
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin).expanduser().resolve()
    return origin if origin.is_file() else None


def mpi4py_libmpi_path() -> Path | None:
    """``libmpi`` shared library linked from the ``mpi4py.MPI`` extension (``ldd``)."""
    ext = mpi4py_mpi_extension_path()
    if ext is None:
        return None
    paths = _libmpi_paths_from_ldd(_run_ldd(ext))
    return paths[0] if paths else None


def mpi4py_openmpi_mismatch() -> tuple[bool, str]:
    """Return ``(ok, message)``; ``ok=False`` when mpi4py and CHARMM use different ``libmpi``."""
    if not _mpi4py_available() or not charmm_lib_links_mpi():
        return True, ""
    charm_mpi = charmm_libmpi_path()
    py_mpi = mpi4py_libmpi_path()
    if charm_mpi is None or py_mpi is None:
        return True, ""
    try:
        same = charm_mpi.resolve() == py_mpi.resolve()
    except OSError:
        same = str(charm_mpi) == str(py_mpi)
    if same:
        return True, ""
    return (
        False,
        "mpi4py is linked to "
        f"{py_mpi} but libcharmm.so uses {charm_mpi}. "
        "MPI_Init will segfault (mixed distro + custom OpenMPI). "
        "Rebuild: ./scripts/rebuild_mpi4py_for_charmm.sh",
    )


def rebuild_mpi4py_shell_hint() -> str:
    mpirun = charmm_mpirun_path()
    if mpirun is None:
        return "./scripts/rebuild_mpi4py_for_charmm.sh"
    bindir = mpirun.parent
    return (
        f"export MPICC={bindir}/mpicc\n"
        f"export MPICXX={bindir}/mpicxx\n"
        "./scripts/rebuild_mpi4py_for_charmm.sh"
    )


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
        mismatch_ok, mismatch_msg = mpi4py_openmpi_mismatch()
        if not mismatch_ok:
            lines.append(f"  mpi4py/OpenMPI mismatch: {mismatch_msg}")
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
    mpi_openmpi_install_env_defaults()
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
    if not _truthy("MMML_NO_MPI_MCA_PREFIX"):
        lines.append("export OMPI_MCA_pmix='^ext3x'")
        mca_dir = openmpi_mca_component_dir()
        if mca_dir is not None:
            lines.append(f"export OMPI_MCA_component_path={mca_dir}")
            lines.append(f"export OMPI_MCA_mca_base_component_path={mca_dir}")
        shmem = openmpi_shmem_mca_for_launch()
        if shmem is not None:
            lines.append(f"export OMPI_MCA_shmem={shmem}")
        opal = (os.environ.get("MMML_OPAL_PREFIX") or "").strip()
        if not opal:
            prefix = openmpi_install_prefix()
            if prefix is not None and (prefix / "share" / "openmpi").is_dir():
                opal = str(prefix)
        if opal:
            lines.append(f"export OPAL_PREFIX={opal}")
    ld = mpi_library_path_export()
    if ld:
        lines.append(ld)
    if not _truthy("MMML_NO_MPI_LD_PATH"):
        preload_var = "DYLD_INSERT_LIBRARIES" if _IS_DARWIN else "LD_PRELOAD"
        preload = (os.environ.get(preload_var) or "").strip()
        if preload:
            lines.append(f"export {preload_var}={preload}")
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


def _preload_openmpi_opal_global() -> None:
    """Preload matched ``libopen-pal`` so distro MCA DSOs resolve OPAL symbols."""
    global _opal_preloaded
    if _opal_preloaded or _truthy("MMML_NO_MPI_LD_PATH") or _truthy("MMML_NO_MPI_OPAL_PRELOAD"):
        return
    opal = openmpi_opal_library_path()
    if opal is None:
        return
    try:
        ctypes.CDLL(str(opal), mode=ctypes.RTLD_GLOBAL)
        _opal_preloaded = True
    except OSError:
        return


def _preload_openmpi_mpi_libraries_global() -> None:
    """``RTLD_GLOBAL`` preload so ``libcharmm.so`` resolves MPI without relying on late ``LD_LIBRARY_PATH``."""
    global _mpi_libs_preloaded
    if _mpi_libs_preloaded or _truthy("MMML_NO_MPI_LD_PATH"):
        return
    candidates = openmpi_mpi_library_paths_for_preload()
    loaded: set[Path] = set()
    for _ in range(max(1, len(candidates))):
        progress = False
        for path in candidates:
            if path in loaded:
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.add(path)
                progress = True
            except OSError:
                continue
        if not progress:
            break
    if loaded:
        _mpi_libs_preloaded = True


def prepare_charmm_mpi_runtime() -> None:
    """Apply MPI/PMIx library path and preload before ``import pycharmm``."""
    _apply_cuda_mpi_env_defaults()
    _scrub_deprecated_openmpi_mca_env()
    if not charmm_lib_links_mpi():
        return
    ensure_charmm_mpi_library_path()
    _preload_openmpi_mpi_libraries_global()
    _preload_openmpi_opal_global()
    _preload_pmix_global()
    try:
        from mmml.utils.jax_gpu_warmup import maybe_sanitize_process_env_for_ptxas

        maybe_sanitize_process_env_for_ptxas(force=True)
    except ImportError:
        pass


def _pin_charmm_openmp_for_serial_mlpot() -> None:
    """Serial MLpot + MPI-linked ``libcharmm.so``: choose OpenMP thread caps.

    Default to one CHARMM/OpenMP thread for conservative MPI-linked MLpot runs.
    When ``MMML_CHARMM_OMP_THREADS`` is explicit, use that same value as the
    default CPU thread budget for BLAS/NumExpr too, unless those variables were
    already set by the caller.
    """
    if _truthy("MMML_NO_CHARMM_OMP_PIN") or not charmm_lib_links_mpi():
        return
    explicit = (os.environ.get("MMML_CHARMM_OMP_THREADS") or "").strip()
    threads = explicit or "1"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ.setdefault("OMP_PROC_BIND", "true")
    cpu_threads = threads if explicit else "1"
    os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpu_threads)
    if explicit:
        os.environ.setdefault("MMML_JAX_COMPILE_THREADS", cpu_threads)
        os.environ["MMML_NO_JAX_COMPILE_THREADS"] = "0"


def configure_mpi4py_charmm_owned_init() -> None:
    """Prevent mpi4py ``MPI_Init`` when CHARMM Fortran owns MPI (idempotent)."""
    global _mpi4py_charmm_configured
    if _mpi4py_charmm_configured:
        return
    if _truthy("MMML_MPI_PY_INIT"):
        return
    if not charmm_lib_links_mpi():
        return
    os.environ.setdefault("MPI4PY_RC_INITIALIZE", "false")
    os.environ.setdefault("MPI4PY_RC_FINALIZE", "false")
    if _truthy("MMML_DEFER_MPI4PY_PACKAGE_IMPORT"):
        _mpi4py_charmm_configured = True
        return
    try:
        import mpi4py

        mpi4py.rc(initialize=False, finalize=False)
        _mpi4py_charmm_configured = True
    except Exception:
        return


def _purge_mpi4py_modules() -> None:
    """Drop cached ``mpi4py`` modules so ``mpi4py.MPI`` can reload after CHARMM init."""
    import sys

    for key in list(sys.modules):
        if key == "mpi4py" or key.startswith("mpi4py."):
            del sys.modules[key]


def ensure_charmm_mpi_initialized() -> None:
    """Load PyCHARMM once so Fortran ``MPI_Init`` runs before mpi4py collectives."""
    global _charmm_mpi_bootstrapped
    if _charmm_mpi_bootstrapped:
        return
    if not charmm_lib_links_mpi():
        return
    configure_mpi4py_charmm_owned_init()
    if not charmm_lib_available():
        return
    import mmml.interfaces.pycharmmInterface.import_pycharmm as import_pycharmm

    import_pycharmm.init_vacuum_charmm_state_mpi()
    _charmm_mpi_bootstrapped = True


def ensure_mpi4py_after_charmm_init(*, phase: str = "after PyCHARMM import") -> bool:
    """Load ``mpi4py.MPI`` after CHARMM Fortran ``MPI_Init``.

    ``configure_mpi4py_charmm_owned_init()`` sets ``mpi4py.rc(initialize=False)``.
    ASE imports ``ase.parallel`` at package load time, which immediately tries
    ``from mpi4py import MPI`` when the ``mpi4py`` package is already present.
    That import must happen **after** CHARMM has initialized MPI, otherwise ASE
    (and any later mpi4py collectives) fail with ``cannot import name 'MPI'``.

    When ``MMML_DEFER_MPI4PY_PACKAGE_IMPORT=1``, the mpi4py package was not
    imported before PyCHARMM; this function performs the first load with MPI
    library paths from ``prepare_charmm_mpi_runtime()`` already applied.
    """
    if _truthy("MMML_MPI_PY_INIT") or not charmm_lib_links_mpi():
        return True
    if not _mpi4py_available():
        return True
    prepare_charmm_mpi_runtime()
    _purge_mpi4py_modules()
    configure_mpi4py_charmm_owned_init()
    try:
        import mpi4py

        mpi4py.rc(initialize=False, finalize=False)
        from mpi4py import MPI

        if not MPI.Is_initialized():
            print(
                f"mmml: mpi4py reports MPI not initialized {phase}. "
                "PyCHARMM should have called MPI_Init from Fortran.",
                file=sys.stderr,
                flush=True,
            )
            return False
        _ = int(MPI.COMM_WORLD.Get_size())
        return True
    except Exception as exc:
        cause = exc.__cause__
        detail = f"{exc}" + (f" (cause: {cause})" if cause else "")
        mismatch_ok, mismatch_msg = mpi4py_openmpi_mismatch()
        hint = "./scripts/rebuild_mpi4py_for_charmm.sh"
        if not mismatch_ok:
            hint = f"{hint}  ({mismatch_msg})"
        print(
            f"mmml: mpi4py.MPI unavailable {phase}: {detail}. "
            f"Rebuild mpi4py against libcharmm OpenMPI: {hint}",
            file=sys.stderr,
            flush=True,
        )
        return False


def mpi_charmm_script(
    script: str,
    *,
    relaxed_bomlev: bool = False,
    quiet: bool = False,
    rank0_drive: bool = False,
    barriers: str = "none",
) -> bool:
    """Run ``lingo.charmm_script`` under MPI.

    Default ``barriers="none"`` matches the workshop pattern: caller synchronises
    ranks in Python, then every rank enters ``eval_charmm_script`` without nested
    mpi4py barriers (those can desync CHARMM's Fortran MPI worker loop).

    By default (**``rank0_drive=False``**) every rank calls ``eval_charmm_script``
    with the same script.  Rank 0 performs disk I/O; Fortran MPI broadcasts state.

    Returns ``True`` when ``lingo.charmm_script`` reports success.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    if size <= 1:
        return _invoke_charmm_script(
            script,
            relaxed_bomlev=relaxed_bomlev,
            quiet=quiet,
        )

    if barriers in ("both", "pre"):
        _mpi_script_barrier()
    ok = True
    try:
        if rank0_drive:
            if rank == 0:
                ok = _invoke_charmm_script(
                    script,
                    relaxed_bomlev=relaxed_bomlev,
                    quiet=quiet,
                )
        else:
            ok = _invoke_charmm_script(
                script,
                relaxed_bomlev=relaxed_bomlev,
                quiet=quiet,
            )
    finally:
        if barriers in ("both", "post"):
            _mpi_script_barrier()
    return ok


def _mpi_script_barrier() -> None:
    if _under_mpirun() and not _mpi_comm_valid():
        ensure_mpi4py_after_charmm_init(phase="mpi_charmm_script barrier")
    if not _mpi_comm_valid():
        return
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()


def _invoke_charmm_script(
    script: str,
    *,
    relaxed_bomlev: bool = False,
    quiet: bool = False,
) -> bool:
    from contextlib import nullcontext

    import pycharmm.lingo as lingo

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        charmm_quiet_output,
        charmm_relaxed_bomlev,
    )

    if relaxed_bomlev:
        ctx = charmm_relaxed_bomlev()
    elif quiet:
        ctx = charmm_quiet_output()
    else:
        ctx = nullcontext()
    with ctx:
        return bool(lingo.charmm_script(script))


def configure_mpi_bootstrap_env() -> None:
    """Env guards before cooperative topology READ (no import-time crystal free).

    Always skip import-time ``reset_block`` for bootstrap: it runs before
    ``bomlev -2`` is applied and aborts on an empty system (BLOCK ILLEGAL COMMAND).
    Under ``mpirun``, import-time BLOCK can also hang on MPI-linked builds.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    _, size = mpi_rank_size()
    os.environ.setdefault("MMML_SKIP_CHARMM_RESET_BLOCK", "1")
    if size <= 1:
        return
    os.environ.setdefault("MMML_DEFER_MPI4PY_PACKAGE_IMPORT", "1")
    os.environ.setdefault("MMML_QUIET", "1")


def sync_import_pycharmm_for_bootstrap(*, tag: str = "bootstrap") -> None:
    """Load ``import_pycharmm`` on all ranks without pre-import mpi4py barriers."""
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    if "mmml.interfaces.pycharmmInterface.import_pycharmm" in sys.modules:
        if size > 1:
            ensure_mpi4py_after_charmm_init(phase=f"{tag} import_pycharmm already loaded")
        return
    if size <= 1:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        return
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    if not ensure_mpi4py_after_charmm_init(phase=f"{tag} synchronized import_pycharmm"):
        raise RuntimeError(f"rank {rank}/{size}: mpi4py.MPI unavailable after import_pycharmm")


def charmm_natom_count() -> int:
    """Return CHARMM PSF atom count (0 when empty)."""
    import pycharmm.psf as psf

    return int(psf.get_natom())


def charmm_natom_diagnostics() -> dict[str, int | bool]:
    """PSF/coor atom counts and whether Fortran reports a loaded PSF."""
    import pycharmm.coor as coor
    import pycharmm.psf as psf

    psf_n = int(psf.get_natom())
    coor_n = int(coor.get_natom())
    return {
        "psf_natom": psf_n,
        "coor_natom": coor_n,
        "psf_loaded": bool(psf_n > 0 or coor_n > 0),
    }


def _psf_atom_types_from_path(psf_path: Path) -> set[str]:
    types: set[str] = set()
    in_atoms = False
    remaining = 0
    for line in psf_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "!NATOM" in line:
            remaining = int(line.split()[0])
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if remaining <= 0:
            break
        parts = line.split()
        if len(parts) >= 6:
            types.add(parts[5])
            remaining -= 1
    if not types:
        raise ValueError(f"No atom types parsed from PSF: {psf_path}")
    return types


def write_minimal_mass_rtf(psf_path: Path, prm_path: Path, out_path: Path) -> Path:
    """Write a MASS-only RTF for atom types in *psf_path*."""
    needed = _psf_atom_types_from_path(psf_path)
    mass_lines: list[str] = []
    seen: set[str] = set()
    for line in prm_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "MASS" and parts[2] in needed:
            mass_lines.append(line)
            seen.add(parts[2])
    missing = sorted(needed - seen)
    if missing:
        raise ValueError(f"Missing MASS records in {prm_path}: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "* MMML MPI bootstrap minimal MASS topology",
        "*",
        *mass_lines,
        "END",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path.resolve()


def ensure_shared_minimal_rtf(psf_path: Path, prm_path: Path) -> Path:
    """Write a MASS-only RTF beside *psf_path* (same path on every MPI rank)."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    out = psf_path.with_name(f"{psf_path.stem}.minimal_mass.rtf")
    rank, size = mpi_rank_size()
    if rank == 0 or (size <= 1 and not out.is_file()):
        write_minimal_mass_rtf(psf_path, prm_path, out)
    if size > 1 and _mpi_comm_valid():
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    return out


def _bootstrap_rank_local_staging_enabled(*, size: int) -> bool:
    """Whether each MPI rank copies topology files to a private UUID directory.

    Default **off**: DOMDEC MPI READ is cooperative (rank 0 disk + Fortran broadcast);
    every rank must pass the **same** paths. Opt in with ``MMML_MPI_BOOTSTRAP_RANK_LOCAL=1``
    only for bisect / embarrassingly parallel workloads.
    """
    if size <= 1:
        return False
    flag = os.environ.get("MMML_MPI_BOOTSTRAP_RANK_LOCAL", "").strip().lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return False


def stage_topology_files_for_rank(
    paths: dict[str, Path],
    *,
    rank: int,
    log_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Path]:
    """Copy topology artifacts to ``$TMPDIR/mmml_mpi_bootstrap/rank<R>_<uuid>/``."""
    run_id = uuid.uuid4().hex
    base = Path(tempfile.gettempdir()) / "mmml_mpi_bootstrap" / f"rank{rank}_{run_id}"
    base.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {"staging_dir": base.resolve()}
    for key, src in paths.items():
        src = Path(src).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"bootstrap staging missing {key}: {src}")
        dst = base / src.name
        shutil.copy2(src, dst)
        staged[key] = dst.resolve()
    if log_fn is not None:
        names = ", ".join(f"{k}={v}" for k, v in staged.items() if k != "staging_dir")
        log_fn("stage", f"rank {rank}: dir={base} {names}")
    return staged


def prepare_rank_local_bootstrap_paths(
    *,
    psf: Path,
    crd: Path,
    prm: Path,
    rank: int,
    size: int,
    rtf_path: Path | None = None,
    res: Path | None = None,
    log_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Path]:
    """Stage PSF/CRD/PRM (and optional RTF/res) under a per-rank UUID directory."""
    run_id = uuid.uuid4().hex
    base = Path(tempfile.gettempdir()) / "mmml_mpi_bootstrap" / f"rank{rank}_{run_id}"
    base.mkdir(parents=True, exist_ok=True)

    if rtf_path is not None and Path(rtf_path).is_file():
        minimal = _copy_bootstrap_file(base, Path(rtf_path))
    else:
        minimal = write_minimal_mass_rtf(psf, prm, base / f"{psf.stem}.minimal_mass.rtf")

    staged: dict[str, Path] = {
        "staging_dir": base.resolve(),
        "rtf": minimal,
        "prm": _copy_bootstrap_file(base, prm),
        "psf": _copy_bootstrap_file(base, psf),
        "crd": _copy_bootstrap_file(base, crd),
    }
    if res is not None and Path(res).is_file():
        staged["res"] = _copy_bootstrap_file(base, res)
    if log_fn is not None:
        log_fn(
            "stage",
            f"rank {rank}/{size}: dir={base} "
            f"rtf={staged['rtf']} psf={staged['psf']} crd={staged['crd']}",
        )
    return staged


def _copy_bootstrap_file(base: Path, src: Path) -> Path:
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"bootstrap staging missing file: {src}")
    dst = base / src.name
    if not dst.is_file():
        shutil.copy2(src, dst)
    return dst.resolve()


def _resolve_bootstrap_topology_paths(
    *,
    psf: Path,
    crd: Path,
    prm: Path,
    rank: int,
    size: int,
    rtf_path: Path | None,
    res: Path | None,
    log_fn: Callable[[str, str], None] | None,
) -> dict[str, Path]:
    if _bootstrap_rank_local_staging_enabled(size=size):
        return prepare_rank_local_bootstrap_paths(
            psf=psf,
            crd=crd,
            prm=prm,
            rank=rank,
            size=size,
            rtf_path=rtf_path,
            res=res,
            log_fn=log_fn,
        )
    if rtf_path is None:
        minimal = _resolve_bootstrap_rtf(
            psf=psf, prm=prm, size=size, rtf_path=rtf_path
        )
    else:
        minimal = Path(rtf_path).expanduser().resolve()
    paths = {
        "rtf": minimal,
        "prm": prm,
        "psf": psf,
        "crd": crd,
    }
    if res is not None and Path(res).is_file():
        paths["res"] = Path(res).expanduser().resolve()
    return paths


def _bootstrap_barrier_enabled() -> bool:
    flag = os.environ.get("MMML_MPI_BOOTSTRAP_BARRIER", "").strip().lower()
    return flag in ("1", "true", "yes")


def sync_bootstrap_ranks(
    *,
    log_fn: Callable[[str, str], None] | None = None,
    label: str = "CHARMM READ",
) -> None:
    """Optional mpi4py barrier (bisect only — **not** used during cooperative READ).

    ``mpi4py.MPI.Barrier`` between ``eval_charmm_script`` calls can desync CHARMM's
    Fortran MPI worker loop and leave ``n_atoms=0`` after READ.  Default bootstrap
    uses one multiline ``mpi_charmm_script`` call with no Python barriers instead.
    Opt in with ``MMML_MPI_BOOTSTRAP_BARRIER=1``.
    """
    if not _bootstrap_barrier_enabled():
        return
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    if size <= 1:
        return
    if not ensure_mpi4py_after_charmm_init(phase=f"bootstrap sync ({label})"):
        raise RuntimeError(f"rank {rank}/{size}: mpi4py.MPI unavailable before bootstrap sync")
    if not _mpi_comm_valid():
        raise RuntimeError(f"rank {rank}/{size}: MPI comm invalid before bootstrap sync")
    from mpi4py import MPI

    if log_fn is not None:
        log_fn("bootstrap_sync", f"rank {rank}/{size} barrier ({label})")
    MPI.COMM_WORLD.Barrier()
    if log_fn is not None:
        log_fn("bootstrap_sync", f"rank {rank}/{size} barrier done ({label})")


def align_mpi_ranks_after_import(
    *,
    log_fn: Callable[[str, str], None] | None = None,
    label: str = "after import_pycharmm",
) -> None:
    """One Python barrier so all ranks finish staggered ``import_pycharmm`` before CHARMM I/O."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    if size <= 1:
        return
    if log_fn is not None:
        log_fn("bootstrap_sync", f"rank {rank}/{size} align ({label})")
    _mpi_script_barrier()
    if log_fn is not None:
        log_fn("bootstrap_sync", f"rank {rank}/{size} align done ({label})")


def _bootstrap_full_rtf() -> bool:
    flag = os.environ.get("MMML_MPI_BOOTSTRAP_FULL_RTF", "").strip().lower()
    return flag in ("1", "true", "yes")


def _resolve_bootstrap_rtf(
    *,
    psf: Path,
    prm: Path,
    size: int,
    rtf_path: Path | None,
) -> Path:
    if rtf_path is not None:
        return Path(rtf_path).expanduser().resolve()
    if size <= 1 or _bootstrap_full_rtf():
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF

        return Path(CGENFF_RTF).expanduser().resolve()
    return ensure_shared_minimal_rtf(psf, prm)


def _bootstrap_force_psf_crd() -> bool:
    flag = os.environ.get("MMML_MPI_BOOTSTRAP_FORCE_PSF_CRD", "").strip().lower()
    return flag in ("1", "true", "yes")


def _bootstrap_rank0_drive() -> bool:
    flag = os.environ.get("MMML_MPI_BOOTSTRAP_RANK0_DRIVE", "").strip().lower()
    return flag in ("1", "true", "yes")


def _resolve_bootstrap_mode(
    mode: str,
    *,
    psf: Path,
    restart_path: str | Path | None,
    size: int,
    rank: int,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, Path | None]:
    """Return ``(effective_mode, res_path)``; at ``np>1`` prefer restart when ``.res`` exists."""
    if mode != "psf-crd" or size <= 1 or _bootstrap_force_psf_crd():
        if mode == "restart":
            res = Path(restart_path or psf.with_suffix(".res")).expanduser().resolve()
            return mode, res
        return mode, None

    res = Path(restart_path or psf.with_suffix(".res")).expanduser().resolve()
    if res.is_file():
        if log_fn is not None:
            log_fn(
                "bootstrap",
                f"rank {rank}/{size}: np>1 psf-crd → restart ({res})",
            )
        return "restart", res

    if log_fn is not None:
        log_fn(
            "bootstrap",
            f"rank {rank}/{size}: np>1 psf-crd (no restart at {res}; "
            "cooperative PSF/CRD READ often leaves n_atoms=0 on DOMDEC MPI builds)",
        )
    return mode, None


def _wait_for_shared_file(path: Path, *, timeout_s: float = 120.0) -> None:
    """Poll until *path* exists with stable non-zero size (rank-0 write handshake)."""
    deadline = time.monotonic() + timeout_s
    last_size = -1
    while time.monotonic() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                return
            last_size = size
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for shared file: {path}")


def _cooperative_read_script(
    paths: dict[str, Path],
    *,
    crystal_side_A: float | None,
    crystal_cutnb_A: float,
) -> str:
    """Multiline READ chain for one barrier-free ``mpi_charmm_script`` call."""
    minimal = paths["rtf"]
    prm = paths["prm"]
    psf = paths["psf"]
    crd = paths["crd"]
    lines = [
        f"read rtf card name {minimal}",
        f"read param card name {prm} flex",
        f"read psf card name {psf}",
        f"read coor card name {crd}",
    ]
    if crystal_side_A is not None:
        side = float(crystal_side_A)
        lines.extend(
            [
                f"crystal define cubic {side} {side} {side} 90 90 90",
                f"crystal build cutoff {float(crystal_cutnb_A)} noper 0",
            ]
        )
    return "\n".join(lines) + "\n"


def _cooperative_restart_script(
    paths: dict[str, Path],
    *,
    fortran_path: str,
    crystal_side_A: float | None,
    crystal_cutnb_A: float,
) -> str:
    """RTF/PRM/PSF then ``read restart`` (coords); no ``UPDATE`` (nbonds not ready yet)."""
    minimal = paths["rtf"]
    prm = paths["prm"]
    psf = paths["psf"]
    lines = [
        f"read rtf card name {minimal}",
        f"read param card name {prm} flex",
        f"read psf card name {psf}",
        f"open read unit 20 name {fortran_path}",
        "read restart unit 20",
        "close unit 20",
    ]
    if crystal_side_A is not None:
        side = float(crystal_side_A)
        lines.extend(
            [
                f"crystal define cubic {side} {side} {side} 90 90 90",
                f"crystal build cutoff {float(crystal_cutnb_A)} noper 0",
            ]
        )
    return "\n".join(lines) + "\n"


# Bump when cooperative np>1 bootstrap I/O strategy changes (read-gate diagnostics).
BOOTSTRAP_MPI_API = "stream-cooperative-v2"


def _cooperative_stream_inp_path(psf: Path, step: str) -> Path:
    return psf.parent / "mpi_bootstrap_stream" / f"{psf.stem}_{step}.inp"


def _run_cooperative_stream_bootstrap(
    step: str,
    paths: dict[str, Path],
    lines: list[str],
    *,
    rank: int,
    size: int,
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """One ``stream`` eval (Fortran reads all lines; avoids Python line-split desync)."""
    psf = paths["psf"]
    inp_path = _cooperative_stream_inp_path(psf, step)
    if rank == 0:
        inp_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["* MMML MPI cooperative bootstrap", "bomlev -2"]
        inp_path.write_text(
            "\n".join([*header, *lines]) + "\n",
            encoding="utf-8",
        )
    if size > 1:
        _wait_for_shared_file(inp_path)
    if log_fn is not None:
        drive = " rank0_drive" if _bootstrap_rank0_drive() else ""
        log_fn(step, f"begin rank {rank}/{size}: stream {inp_path}{drive}")
    stream_ok = mpi_charmm_script(
        f"stream {inp_path}\n",
        relaxed_bomlev=True,
        rank0_drive=_bootstrap_rank0_drive(),
    )
    diag = charmm_natom_diagnostics()
    n_atoms = int(diag["psf_natom"])
    if log_fn is not None:
        log_fn(
            step,
            f"done rank {rank}/{size}: n_atoms={n_atoms} "
            f"stream_ok={stream_ok} psf={diag['psf_natom']} coor={diag['coor_natom']}",
        )
    return n_atoms


def _run_cooperative_bootstrap_script(
    step: str,
    script: str,
    *,
    paths: dict[str, Path] | None = None,
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Run CHARMM script on all ranks without mpi4py barriers (workshop pattern)."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    body = [ln for ln in script.strip().splitlines() if ln.strip()]
    if size > 1 and paths is not None and len(body) > 1:
        return _run_cooperative_stream_bootstrap(
            step,
            paths,
            body,
            rank=rank,
            size=size,
            log_fn=log_fn,
        )
    n_lines = max(1, len(body))
    if log_fn is not None:
        log_fn(step, f"begin rank {rank}/{size}: {n_lines} line(s)")
    mpi_charmm_script(script if script.endswith("\n") else script + "\n", relaxed_bomlev=True)
    n_atoms = charmm_natom_count()
    if log_fn is not None:
        log_fn(step, f"done rank {rank}/{size}: n_atoms={n_atoms}")
    return n_atoms


def bootstrap_charmm_step(
    step: str,
    script: str,
    *,
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Run one CHARMM script on all ranks; return ``n_atoms`` after the step."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    if _bootstrap_barrier_enabled():
        sync_bootstrap_ranks(log_fn=log_fn, label=f"before {step}")
    if log_fn is not None:
        log_fn(step, f"begin rank {rank}/{size}: {script.strip()}")
    mpi_charmm_script(script, relaxed_bomlev=True)
    n_atoms = charmm_natom_count()
    if log_fn is not None:
        log_fn(step, f"done rank {rank}/{size}: n_atoms={n_atoms}")
    return n_atoms


def _bootstrap_via_stream_inp(
    paths: dict[str, Path],
    *,
    psf: Path,
    crystal_side_A: float | None,
    crystal_cutnb_A: float,
    rank: int,
    size: int,
    log_fn: Callable[[str, str], None] | None = None,
) -> None:
    """Load topology via one ``stream`` (native CHARMM input path; best for np>1)."""
    lines = _cooperative_read_script(
        paths,
        crystal_side_A=crystal_side_A,
        crystal_cutnb_A=crystal_cutnb_A,
    ).strip().splitlines()
    _run_cooperative_stream_bootstrap(
        "stream_inp",
        paths,
        lines,
        rank=rank,
        size=size,
        log_fn=log_fn,
    )


def bootstrap_topology_mpi(
    psf_path: str | Path,
    crd_path: str | Path,
    *,
    prm_path: str | Path | None = None,
    rtf_path: str | Path | None = None,
    crystal_side_A: float | None = None,
    crystal_cutnb_A: float = 15.0,
    restart_path: str | Path | None = None,
    mode: str = "psf-crd",
    log_fn: Callable[[str, str], None] | None = None,
) -> int:
    """Cooperative ``np>1`` topology bootstrap.

    Supported ``mode`` values: ``psf-crd``, ``stream-inp``, ``restart``.
    At ``np>1``, ``psf-crd`` auto-switches to ``restart`` when a sidecar ``.res``
    exists (from ``--prepare-prebuilt-only``). Set
    ``MMML_MPI_BOOTSTRAP_FORCE_PSF_CRD=1`` to bisect cooperative PSF/CRD READ.
    """
    from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    psf = Path(psf_path).expanduser().resolve()
    crd = Path(crd_path).expanduser().resolve()
    if not psf.is_file():
        raise FileNotFoundError(f"PSF not found: {psf}")
    if mode != "restart" and not crd.is_file():
        raise FileNotFoundError(f"CRD not found: {crd}")

    if prm_path is None:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

        prm = Path(CGENFF_PRM).expanduser().resolve()
    else:
        prm = Path(prm_path).expanduser().resolve()

    rank, size = mpi_rank_size()
    rtf_opt = Path(rtf_path).expanduser().resolve() if rtf_path else None
    effective_mode, res_opt = _resolve_bootstrap_mode(
        mode,
        psf=psf,
        restart_path=restart_path,
        size=size,
        rank=rank,
        log_fn=log_fn,
    )
    if effective_mode == "restart" and (res_opt is None or not res_opt.is_file()):
        raise FileNotFoundError(f"Restart not found: {res_opt}")

    paths = _resolve_bootstrap_topology_paths(
        psf=psf,
        crd=crd,
        prm=prm,
        rank=rank,
        size=size,
        rtf_path=rtf_opt,
        res=res_opt if effective_mode == "restart" else None,
        log_fn=log_fn,
    )
    if size > 1:
        align_mpi_ranks_after_import(log_fn=log_fn)

    if effective_mode == "restart":
        res_local = paths["res"]
        fortran_path, alias = charmm_fortran_path(res_local, for_write=False)
        try:
            restart_script = _cooperative_restart_script(
                paths,
                fortran_path=fortran_path,
                crystal_side_A=crystal_side_A,
                crystal_cutnb_A=crystal_cutnb_A,
            )
            if size > 1:
                _run_cooperative_bootstrap_script(
                    "read_restart",
                    restart_script,
                    paths=paths,
                    log_fn=log_fn,
                )
            else:
                for step_name, script in (
                    ("read_rtf", f"read rtf card name {paths['rtf']}\n"),
                    ("read_prm", f"read param card name {paths['prm']} flex\n"),
                    ("read_psf", f"read psf card name {paths['psf']}\n"),
                    ("open_restart", f"open read unit 20 name {fortran_path}\n"),
                    ("read_restart", "read restart unit 20\n"),
                    ("close_restart", "close unit 20\n"),
                ):
                    bootstrap_charmm_step(step_name, script, log_fn=log_fn)
        finally:
            if alias is not None:
                alias.finalize()
    elif effective_mode == "stream-inp":
        _bootstrap_via_stream_inp(
            paths,
            psf=psf,
            crystal_side_A=crystal_side_A,
            crystal_cutnb_A=crystal_cutnb_A,
            rank=rank,
            size=size,
            log_fn=log_fn,
        )
    elif effective_mode == "psf-crd" and size > 1:
        read_script = _cooperative_read_script(
            paths,
            crystal_side_A=crystal_side_A,
            crystal_cutnb_A=crystal_cutnb_A,
        )
        _run_cooperative_bootstrap_script(
            "read_psf_crd",
            read_script,
            paths=paths,
            log_fn=log_fn,
        )
    else:
        if effective_mode != "psf-crd":
            raise ValueError(f"unsupported bootstrap mode: {mode!r}")
        minimal = paths["rtf"]
        prm = paths["prm"]
        psf = paths["psf"]
        crd = paths["crd"]
        steps = [
            ("read_rtf", f"read rtf card name {minimal}\n"),
            ("read_prm", f"read param card name {prm} flex\n"),
            ("read_psf", f"read psf card name {psf}\n"),
            ("read_coor", f"read coor card name {crd}\n"),
        ]
        if crystal_side_A is not None:
            side = float(crystal_side_A)
            steps.extend(
                [
                    (
                        "crystal_define",
                        f"crystal define cubic {side} {side} {side} 90 90 90\n",
                    ),
                    (
                        "crystal_build",
                        f"crystal build cutoff {float(crystal_cutnb_A)} noper 0\n",
                    ),
                ]
            )
        for step_name, script in steps:
            bootstrap_charmm_step(step_name, script, log_fn=log_fn)

    n_final = charmm_natom_count()
    if n_final <= 0:
        raise RuntimeError(
            f"rank {rank}/{size}: empty CHARMM state after bootstrap "
            f"(requested={mode!r} effective={effective_mode!r})"
        )
    return n_final


def disable_ase_mpi_parallel() -> None:
    """Force ASE serial I/O when mpi4py is loaded for CHARMM (avoids bcast clashes)."""
    try:
        import ase.parallel as ase_parallel
    except ImportError:
        return
    ase_parallel.MPI.comm = ase_parallel.DummyMPI()


def prepare_serial_charmm_mpi_env() -> None:
    """Env/LD setup only — do **not** call ``MPI_Init`` from mpi4py (CHARMM owns that)."""
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        sanitize_xla_flags_env,
    )

    sanitize_xla_flags_env(quiet=True)
    prepare_charmm_mpi_runtime()
    if charmm_lib_links_mpi():
        os.environ.setdefault("MMML_NO_JAX_COMPILE_THREADS", "1")
        configure_mpi4py_charmm_owned_init()
    _pin_charmm_openmp_for_serial_mlpot()
    if _under_mpirun():
        from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
            pin_cuda_for_spatial_mpi,
        )

        pin_cuda_for_spatial_mpi()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
    mpi_openmpi_install_env_defaults()
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
