"""Resolve ``CHARMM_HOME`` / ``CHARMM_LIB_DIR`` for PyCHARMM."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_CGENFF_RTF_NAME = "top_all36_cgenff.rtf"
_CGENFF_PRM_NAME = "par_all36_cgenff.prm"
# Bundled CGENFF toppar shipped in git (bytes); catches empty/partial checkouts.
_MIN_CGENFF_RTF_BYTES = 1_000_000
_MIN_CGENFF_PRM_BYTES = 500_000

_CHARMM_LIB_NAMES = ("libcharmm.so", "libcharmm.dylib", "charmm.so", "charmm.dylib")


def mmml_repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "mmml").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


def find_charmm_lib_in_dir(lib_dir: Path) -> Path | None:
    """Return the first shared library under *lib_dir* or ``lib_dir/lib``."""
    for name in _CHARMM_LIB_NAMES:
        candidate = lib_dir / name
        if candidate.is_file():
            return candidate
    lib_subdir = lib_dir / "lib"
    for name in _CHARMM_LIB_NAMES:
        candidate = lib_subdir / name
        if candidate.is_file():
            return candidate
    return None


def default_repo_charmm_home(repo_root: Path | None = None) -> Path | None:
    """``setup/charmm`` when a ``libcharmm`` shared library is present there."""
    root = repo_root or mmml_repo_root()
    candidate = root / "setup" / "charmm"
    if find_charmm_lib_in_dir(candidate):
        return candidate
    return None


def charmm_build_cache_dirs(env: "os._Environ | dict[str, str] | None" = None) -> list[Path]:
    """Out-of-tree build directories that hold a ``libcharmm``.

    ``scripts/rebuild_charmm_mlpot.sh`` builds into
    ``$HOME/.cache/mmml-charmm-build/<platform-tag>`` (overridable with
    ``CHARMM_BUILD_DIR``), and the per-tier helpers add
    ``.../tier_<max_npr>_nodomdec/lib``. Those builds are frequently *newer*
    than the copy under ``setup/charmm``, so they must be discoverable — a
    stale in-tree library silently caps MLpot at the conservative
    ``max_Nml``/``max_Npr`` fallback even when a fresh build exists.

    Reads ``CHARMM_BUILD_DIR`` / ``HOME`` from *env* (default ``os.environ``);
    passing an explicit mapping keeps discovery hermetic under test.
    """
    environ = env if env is not None else os.environ
    roots: list[Path] = []
    explicit = (environ.get("CHARMM_BUILD_DIR") or "").strip()
    if explicit:
        build_dir = Path(explicit).expanduser()
        # CHARMM_BUILD_DIR names one build directory; its siblings are the
        # other platform/tier builds from the same cache.
        roots.extend([build_dir, build_dir.parent])
    home_raw = (environ.get("HOME") or "").strip()
    if home_raw:
        roots.append(Path(home_raw) / ".cache" / "mmml-charmm-build")
    elif env is None:
        roots.append(Path("~/.cache/mmml-charmm-build").expanduser())

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root, *(c for c in sorted(root.iterdir()) if c.is_dir())]
        for candidate in candidates:
            if find_charmm_lib_in_dir(candidate) is None:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def newest_charmm_lib_dir(candidates: list[Path]) -> Path | None:
    """The candidate whose ``libcharmm`` has the most recent mtime."""
    best: Path | None = None
    best_mtime: float | None = None
    for directory in candidates:
        lib = find_charmm_lib_in_dir(directory)
        if lib is None:
            continue
        mtime = lib.stat().st_mtime
        if best_mtime is None or mtime > best_mtime:
            best, best_mtime = directory, mtime
    return best


def normalize_charmm_lib_dir(raw: str | None) -> str:
    """Return a directory path for ``CHARMM_LIB_DIR`` (not a ``.so`` file path)."""
    value = (raw or "").strip()
    if not value:
        return ""
    path = Path(value)
    if path.suffix in (".so", ".dylib") and path.is_file():
        return str(path.parent)
    return value


def _resolve_lib_dir(
    *,
    env: os._Environ,
    default: str,
) -> str:
    """Pick the first ``CHARMM_LIB_DIR`` candidate that contains ``libcharmm``."""
    candidates: list[str] = []
    for raw in (env.get("CHARMM_LIB_DIR"), default):
        norm = normalize_charmm_lib_dir(raw)
        if norm and norm not in candidates:
            candidates.append(norm)
    for candidate in candidates:
        if find_charmm_lib_in_dir(Path(candidate)):
            return candidate
    return candidates[0] if candidates else ""


def _resolve_one(
    key: str,
    *,
    env: os._Environ,
    default: str,
) -> str:
    explicit = (env.get(key) or "").strip()
    if explicit:
        return explicit
    return default


def _valid_charmm_home(path: str) -> bool:
    """True when *path* exists (directory) or contains a ``libcharmm`` shared lib."""
    if not path:
        return False
    p = Path(path)
    if p.is_dir():
        return True
    return find_charmm_lib_in_dir(p) is not None


def resolve_charmm_paths(
    *,
    repo_root: Path | None = None,
    env: os._Environ | None = None,
) -> tuple[str, str]:
    """Return ``(CHARMM_HOME, CHARMM_LIB_DIR)``.

    Precedence per variable: explicit environment → repo ``setup/charmm`` when
    ``libcharmm`` is present there.

    Neither variable needs to be set: a repo built with ``make install-native``
    is discovered automatically. Set them only to point at an out-of-tree CHARMM
    (for example the per-tier libs built by ``ensure_charmm_mlpot_limits.sh``).
    """
    environ = env if env is not None else os.environ
    root = repo_root or mmml_repo_root()

    default_home = default_repo_charmm_home(root)
    default_home_s = str(default_home) if default_home else ""

    home = _resolve_one("CHARMM_HOME", env=environ, default=default_home_s)
    if home and not _valid_charmm_home(home) and default_home_s:
        home = default_home_s

    # CHARMM_HOME stays the *source* tree (it owns source/api/api_func.F90 and
    # toppar); only the library directory follows the freshest build. Prefer an
    # out-of-tree build-cache library over a stale setup/charmm copy, matching
    # what the cluster workflow scripts set by hand.
    default_lib_s = default_home_s
    freshest = newest_charmm_lib_dir(
        [
            *([default_home] if default_home else []),
            *charmm_build_cache_dirs(env=environ),
        ]
    )
    if freshest is not None:
        default_lib_s = str(freshest)

    lib = _resolve_lib_dir(env=environ, default=default_lib_s)
    if lib and not _valid_charmm_home(lib) and default_lib_s:
        lib = default_lib_s

    return home, lib


def bootstrap_charmm_env(
    *,
    repo_root: Path | None = None,
    env: os._Environ | None = None,
) -> tuple[str, str]:
    """Apply the discovery chain via ``setdefault`` on the target environment."""
    environ = env if env is not None else os.environ
    home, lib = resolve_charmm_paths(repo_root=repo_root, env=environ)
    if home:
        environ.setdefault("CHARMM_HOME", home)
    if lib:
        environ.setdefault("CHARMM_LIB_DIR", lib)
    return home, lib


def _charmm_io_aliases_disabled() -> bool:
    raw = (os.environ.get("MMML_CHARMM_IO_ALIASES") or "1").strip().lower()
    return raw in ("0", "false", "no", "off")


def _path_component_has_uppercase(part: str) -> bool:
    """True when a path segment contains uppercase (ignores single-letter tokens like ``T``)."""
    if part in (".", ".."):
        return False
    if len(part) == 1 and part.isalpha():
        return False
    return any(ch.isupper() for ch in part)


def charmm_fortran_max_path_length() -> int:
    """CHARMM Fortran ``OPEN``/``WRITE`` name buffer (typically 128 characters)."""
    raw = (os.environ.get("MMML_CHARMM_MAX_PATH_LEN") or "").strip()
    if raw:
        return max(64, int(raw))
    return 128


def fortran_path_needs_alias(path: str | Path, *, for_write: bool = False) -> bool:
    """True when CHARMM Fortran I/O may fail on *path* (uppercase, long, or MPI)."""
    if _charmm_io_aliases_disabled():
        return False
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    resolved = p.resolve()
    # Library-mode CHARMM Fortran OPEN is case-sensitive; always stage writes to a
    # lowercase path under $TMPDIR/mmml-charmm-io and copy back afterward.
    if for_write:
        return True
    if len(str(resolved)) > charmm_fortran_max_path_length():
        return True
    if any(_path_component_has_uppercase(part) for part in resolved.parts):
        return True
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        if _under_mpirun() and charmm_lib_links_mpi():
            return True
    except ImportError:
        pass
    return False


def charmm_io_staging_root() -> Path:
    raw = (os.environ.get("MMML_CHARMM_IO_STAGING") or "").strip()
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    base = Path(os.environ.get("TMPDIR", "/tmp"))
    # Per-user directory name: on shared compute nodes a legacy flat
    # ``/tmp/mmml-charmm-io`` is often owned by another account (mode 755),
    # which blocks mkdir for everyone else.
    user = (os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()
    if not user:
        user = f"u{os.getuid()}"
    return base / f"mmml-charmm-io-{user}"


def _charmm_io_alias_scope() -> str:
    """Isolate staging aliases per job/process (shared ``/tmp`` on compute nodes)."""
    for key in ("SLURM_JOB_ID", "MMML_CHARMM_IO_SCOPE"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    return str(os.getpid())


def _ensure_read_symlink(alias: Path, original: Path) -> None:
    """Create or reuse a read symlink; tolerate concurrent creators on shared ``/tmp``."""
    target = original.resolve()
    if alias.is_symlink():
        try:
            if alias.resolve() == target:
                return
        except OSError:
            pass
        alias.unlink()
    elif alias.exists():
        alias.unlink()
    try:
        alias.symlink_to(target)
    except FileExistsError:
        if alias.is_symlink():
            try:
                if alias.resolve() == target:
                    return
            except OSError:
                pass
        raise


@dataclass
class CharmmIoAlias:
    """Lowercase staging path for CHARMM ``OPEN`` when the real path has capitals."""

    original: Path
    alias: Path
    for_write: bool
    _finalized: bool = field(default=False, repr=False)

    @property
    def fortran_path(self) -> str:
        return str(self.alias)

    def finalize(self) -> None:
        """After a write, copy the staging file back to ``original``."""
        if self._finalized:
            return
        self._finalized = True
        if not self.for_write or not self.alias.is_file():
            return
        self.original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.alias, self.original)


def _charmm_io_staging_alias_path(
    path: str | Path,
    *,
    for_write: bool = False,
    staging_root: Path | None = None,
) -> Path | None:
    """Lowercase Fortran staging path for *path*, or ``None`` when staging is off."""
    original = Path(path).expanduser().resolve()
    if not fortran_path_needs_alias(original, for_write=for_write):
        return None
    root = staging_root or charmm_io_staging_root()
    scope = _charmm_io_alias_scope()
    tag = hashlib.sha256(f"{original.resolve()}|{scope}".encode()).hexdigest()[:16]
    return root / tag / original.name.lower()


def remove_charmm_io_write_staging_alias(
    path: str | Path,
    *,
    staging_root: Path | None = None,
) -> bool:
    """Delete a staged write alias so the next DCD open starts from an empty file.

    ``_reset_stage_trajectory`` only removes the real output path; aborted runs can
    leave a partial binary DCD under ``$TMPDIR/mmml-charmm-io``.  Reopening that
    alias via ``dynamics_set_iuncrd`` then triggers formatted/unformatted READ errors
    in ``dynio.F90``.
    """
    alias = _charmm_io_staging_alias_path(
        path,
        for_write=True,
        staging_root=staging_root,
    )
    if alias is None or not alias.is_file():
        return False
    alias.unlink()
    return True


def charmm_io_alias(
    path: str | Path,
    *,
    for_write: bool = False,
    append: bool = False,
    staging_root: Path | None = None,
) -> CharmmIoAlias | None:
    """Return a lowercase alias when CHARMM cannot open *path* directly."""
    original = Path(path).expanduser().resolve()
    alias_path = _charmm_io_staging_alias_path(
        original,
        for_write=for_write,
        staging_root=staging_root,
    )
    if alias_path is None:
        return None

    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias = alias_path

    if for_write:
        if append and original.is_file() and not alias.is_file():
            shutil.copy2(original, alias)
        elif not append and alias.is_file():
            alias.unlink()
    else:
        if not original.is_file():
            raise FileNotFoundError(f"restart not found: {original}")
        _ensure_read_symlink(alias, original)

    return CharmmIoAlias(original=original, alias=alias, for_write=for_write)


def charmm_fortran_path(
    path: str | Path,
    *,
    for_write: bool = False,
    append: bool = False,
    staging_root: Path | None = None,
) -> tuple[str, CharmmIoAlias | None]:
    """Return ``(path_for_charmm, alias_or_none)``."""
    alias = charmm_io_alias(
        path,
        for_write=for_write,
        append=append,
        staging_root=staging_root,
    )
    if alias is None:
        return str(Path(path).expanduser().resolve()), None
    return alias.fortran_path, alias


@dataclass(frozen=True)
class CgenffTopparPaths:
    """Resolved CGENFF RTF/PRM pair under the MMML repo."""

    rtf: Path
    prm: Path


def _cgenff_toppar_search_dirs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "mmml" / "data" / "charmm",
        repo_root / "setup" / "charmm" / "toppar",
    ]


def resolve_cgenff_toppar_paths(*, repo_root: Path | None = None) -> CgenffTopparPaths:
    """Locate bundled CGENFF toppar (``mmml/data/charmm`` first, then ``setup/charmm/toppar``)."""
    root = repo_root or mmml_repo_root()
    for base in _cgenff_toppar_search_dirs(root):
        rtf = base / _CGENFF_RTF_NAME
        prm = base / _CGENFF_PRM_NAME
        if rtf.is_file() and prm.is_file():
            return CgenffTopparPaths(rtf=rtf.resolve(), prm=prm.resolve())
    tried = "\n".join(
        f"  {base / _CGENFF_RTF_NAME} ({'found' if (base / _CGENFF_RTF_NAME).is_file() else 'missing'}), "
        f"{base / _CGENFF_PRM_NAME} ({'found' if (base / _CGENFF_PRM_NAME).is_file() else 'missing'})"
        for base in _cgenff_toppar_search_dirs(root)
    )
    raise FileNotFoundError(
        "CGENFF toppar not found. Expected both "
        f"{_CGENFF_RTF_NAME!r} and {_CGENFF_PRM_NAME!r} in the repo.\n"
        f"Searched:\n{tried}\n"
        "Run `git pull` (or `git lfs pull`) on the cluster checkout."
    )


def assert_cgenff_toppar_readable(
    paths: CgenffTopparPaths | None = None,
    *,
    repo_root: Path | None = None,
) -> CgenffTopparPaths:
    """Fail fast before CHARMM ``read_param_file`` with a misleading RTF I/O message."""
    resolved = paths or resolve_cgenff_toppar_paths(repo_root=repo_root)
    issues: list[str] = []
    for label, path, min_bytes in (
        ("RTF", resolved.rtf, _MIN_CGENFF_RTF_BYTES),
        ("PRM", resolved.prm, _MIN_CGENFF_PRM_BYTES),
    ):
        if not path.is_file():
            issues.append(f"CGENFF {label} missing: {path}")
            continue
        size = path.stat().st_size
        if size < min_bytes:
            issues.append(
                f"CGENFF {label} too small ({size} bytes, expected >= {min_bytes}): {path}"
            )
    if issues:
        raise FileNotFoundError(
            "CGENFF toppar is missing or incomplete on disk.\n"
            + "\n".join(f"  - {line}" for line in issues)
            + "\nCHARMM may report `read_param_file: io error opening/closing rtf file` "
            "when the PRM open/read fails."
        )
    return resolved
