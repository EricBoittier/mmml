"""Unit-test fixtures (no libcharmm.so required)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest


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
