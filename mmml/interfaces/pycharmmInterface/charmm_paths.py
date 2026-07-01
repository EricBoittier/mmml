"""Resolve ``CHARMM_HOME`` / ``CHARMM_LIB_DIR`` for PyCHARMM."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_CHARMM_LIB_NAMES = ("libcharmm.so", "libcharmm.dylib", "charmm.so", "charmm.dylib")
_CHARMMSETUP_KEYS = frozenset({"CHARMM_HOME", "CHARMM_LIB_DIR"})


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


def read_charmmsetup(repo_root: Path | None = None) -> dict[str, str]:
    """Parse legacy ``CHARMMSETUP`` (``export KEY=val`` or ``KEY=val``)."""
    root = repo_root or mmml_repo_root()
    setup_file = root / "CHARMMSETUP"
    out: dict[str, str] = {}
    if not setup_file.is_file():
        return out
    for line in setup_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in _CHARMMSETUP_KEYS:
            out[key] = value.strip()
    return out


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
    setup: dict[str, str],
    default: str,
) -> str:
    """Pick the first ``CHARMM_LIB_DIR`` candidate that contains ``libcharmm``."""
    candidates: list[str] = []
    for raw in (
        env.get("CHARMM_LIB_DIR"),
        setup.get("CHARMM_LIB_DIR"),
        default,
    ):
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
    setup: dict[str, str],
    default: str,
) -> str:
    explicit = (env.get(key) or "").strip()
    if explicit:
        return explicit
    from_setup = (setup.get(key) or "").strip()
    if from_setup:
        return from_setup
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

    Precedence per variable: explicit environment → ``CHARMMSETUP`` → repo
    ``setup/charmm`` when ``libcharmm`` is present there.
    """
    environ = env if env is not None else os.environ
    root = repo_root or mmml_repo_root()
    setup = read_charmmsetup(root)

    default_home = default_repo_charmm_home(root)
    default_home_s = str(default_home) if default_home else ""

    home = _resolve_one("CHARMM_HOME", env=environ, setup=setup, default=default_home_s)
    if home and not _valid_charmm_home(home) and default_home_s:
        home = default_home_s

    lib = _resolve_lib_dir(env=environ, setup=setup, default=default_home_s)
    if lib and not _valid_charmm_home(lib) and default_home_s:
        lib = default_home_s

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
    return Path(os.environ.get("TMPDIR", "/tmp")) / "mmml-charmm-io"


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


def charmm_io_alias(
    path: str | Path,
    *,
    for_write: bool = False,
    append: bool = False,
    staging_root: Path | None = None,
) -> CharmmIoAlias | None:
    """Return a lowercase alias when CHARMM cannot open *path* directly."""
    original = Path(path).expanduser().resolve()
    if not fortran_path_needs_alias(original, for_write=for_write):
        return None

    root = staging_root or charmm_io_staging_root()
    tag = hashlib.sha256(str(original).encode()).hexdigest()[:16]
    alias_dir = root / tag
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / original.name.lower()

    if for_write:
        if append and original.is_file() and not alias.is_file():
            shutil.copy2(original, alias)
    else:
        if not original.is_file():
            raise FileNotFoundError(f"restart not found: {original}")
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(original)

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
