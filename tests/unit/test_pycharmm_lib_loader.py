"""Cross-platform resolution of the CHARMM shared library (``pycharmm/lib.py``).

Importing ``pycharmm.lib`` loads CHARMM at module scope, so these tests skip
where ``libcharmm`` is unavailable — matching the repo's convention for
CHARMM-dependent tests. The pure resolver helpers are what we exercise: the
platform suffix map, explicit-dir resolution, and repo ``setup/charmm``
auto-discovery (``.dylib`` on macOS, ``.so`` on Linux).
"""

from __future__ import annotations

import pytest


def _lib():
    try:
        import pycharmm.lib as lib
    except OSError:
        pytest.skip("libcharmm not available in this environment")
    return lib


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
