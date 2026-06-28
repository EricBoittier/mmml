"""Resolve ``CHARMM_HOME`` / ``CHARMM_LIB_DIR`` for PyCHARMM."""

from __future__ import annotations

import os
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

    lib = _resolve_lib_dir(env=environ, setup=setup, default=default_home_s)

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
