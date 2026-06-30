"""Unit-test fixtures (no libcharmm.so required)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

MINIMAL_RESTART_HEADER = (
    "REST     0     1\n"
    " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
    "         2           0           0           0           0           0           0\n"
)


def write_minimal_restart(path: Path, *, content: str | None = None) -> Path:
    """Write a CHARMM restart stub that passes ``_valid_restart_file``."""
    path.write_text(content or MINIMAL_RESTART_HEADER, encoding="utf-8")
    return path


@contextmanager
def _noop_charmm_quiet_output():
    yield


@pytest.fixture(autouse=True)
def mock_charmm_quiet_output_for_unit_tests(monkeypatch):
    """Avoid lazy PyCHARMM import inside charmm_quiet_output on CI without libcharmm."""
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
        _noop_charmm_quiet_output,
    )


@pytest.fixture(autouse=True)
def _clear_stub_charmm_lib_env(monkeypatch):
    """Drop CHARMM_HOME/LIB_DIR when they point at pytest stub libs (file too short)."""
    lib_dir = os.environ.get("CHARMM_LIB_DIR")
    if lib_dir:
        stub = Path(lib_dir)
        lib = stub / "libcharmm.so" if stub.is_dir() else stub
        if lib.is_file() and lib.stat().st_size < 4096:
            monkeypatch.delenv("CHARMM_HOME", raising=False)
            monkeypatch.delenv("CHARMM_LIB_DIR", raising=False)
    yield
