"""Cross-platform resolution of the CHARMM shared library (``pycharmm/lib.py``).

Importing ``pycharmm.lib`` loads CHARMM at module scope, so these tests skip
where ``libcharmm`` is unavailable — matching the repo's convention for
CHARMM-dependent tests. The pure resolver helpers are what we exercise: the
platform suffix map, explicit-dir resolution, and repo ``setup/charmm``
auto-discovery (``.dylib`` on macOS, ``.so`` on Linux).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

# MMML ships its cross-platform resolver in the vendored pycharmm ``lib.py``.
# The active ``import pycharmm`` may resolve to a different CHARMM build that
# lacks these helpers, so exercise the vendored source directly and
# deterministically instead of whatever happens to be installed.
_VENDORED_LIB = (
    Path(__file__).resolve().parents[2]
    / "setup/charmm/tool/pycharmm/pycharmm/lib.py"
)


def _lib():
    if not _VENDORED_LIB.is_file():
        pytest.skip(f"vendored pycharmm lib.py not found at {_VENDORED_LIB}")
    ns: dict = {"__file__": str(_VENDORED_LIB)}
    # The module instantiates ``CharmmLib()`` at import scope, which loads (and
    # would init) the CHARMM shared library — a second, in-process CHARMM load
    # that can hang or crash the test. The resolver helpers we exercise are all
    # defined above that line, so truncate the source before it and exec only
    # the pure-Python definitions.
    source = _VENDORED_LIB.read_text(encoding="utf-8")
    marker = "\ncharmm_lib = CharmmLib("
    if marker in source:
        source = source.split(marker, 1)[0]
    exec(compile(source, str(_VENDORED_LIB), "exec"), ns)
    required = ("charmm_lib_suffix", "resolve_charmm_lib_path", "_discover_repo_charmm_lib")
    missing = [name for name in required if name not in ns]
    if missing:
        pytest.skip(f"vendored pycharmm lib.py missing helpers: {missing}")
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def test_suffix_is_platform_correct():
    lib = _lib()
    assert lib.charmm_lib_suffix("Darwin") == ".dylib"
    assert lib.charmm_lib_suffix("Linux") == ".so"
    assert lib.charmm_lib_suffix("Windows") == ".dll"
    # unknown platforms fall back to the Unix shared-object suffix
    assert lib.charmm_lib_suffix("Plan9") == ".so"


def test_explicit_dir_resolution(tmp_path):
    lib = _lib()
    suffix = lib.charmm_lib_suffix()

    # missing lib -> joined path with the platform suffix (CDLL raises later,
    # against the location the caller asked for)
    resolved = lib.resolve_charmm_lib_path(str(tmp_path))
    assert resolved.endswith("libcharmm" + suffix)

    # a present lib in the dir is found and returned as a full path
    fake = tmp_path / ("libcharmm" + suffix)
    fake.write_bytes(b"")
    assert lib.resolve_charmm_lib_path(str(tmp_path)) == str(fake)


def test_lib_subdir_resolution(tmp_path):
    lib = _lib()
    suffix = lib.charmm_lib_suffix()
    subdir = tmp_path / "lib"
    subdir.mkdir()
    fake = subdir / ("libcharmm" + suffix)
    fake.write_bytes(b"")
    assert lib.resolve_charmm_lib_path(str(tmp_path)) == str(fake)


def test_repo_autodiscovery():
    lib = _lib()
    found = lib._discover_repo_charmm_lib(lib.charmm_lib_suffix())
    # None when no bundled lib, else a path under setup/charmm.
    if found is not None:
        assert "setup" in found and "charmm" in found
        assert found.endswith(lib.charmm_lib_suffix())
