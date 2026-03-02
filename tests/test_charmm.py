"""
Tests for CHARMM / PyCHARMM integration.

CHARMM is optional and requires CHARMM_HOME, CHARMM_LIB_DIR, and the pycharmm
package. These tests verify the interface works when available.
"""

from __future__ import annotations

import os
import pytest


def _can_import_pycharmm() -> bool:
    """Return True if pycharmm can be imported."""
    try:
        __import__("pycharmm")
        return True
    except Exception:
        return False


def _charmm_env_configured() -> bool:
    """Return True if CHARMM env vars are set and paths exist."""
    home = os.environ.get("CHARMM_HOME")
    lib = os.environ.get("CHARMM_LIB_DIR")
    if not home or not lib:
        return False
    return os.path.exists(home) and os.path.exists(lib)


@pytest.mark.skipif(
    not _charmm_env_configured(),
    reason="CHARMM_HOME or CHARMM_LIB_DIR not set or paths do not exist",
)
def test_charmm_import():
    """PyCHARMM can be imported when CHARMM env is configured."""
    import pycharmm  # noqa: F401
    assert pycharmm is not None


@pytest.mark.skipif(
    not _can_import_pycharmm(),
    reason="pycharmm not available in this environment",
)
def test_mmml_calculator_charmm_flag():
    """MMML calculator module exposes _HAVE_PYCHARMM correctly."""
    from mmml.pycharmmInterface import mmml_calculator

    assert hasattr(mmml_calculator, "_HAVE_PYCHARMM")
    assert mmml_calculator._HAVE_PYCHARMM is True


@pytest.mark.skipif(
    not _can_import_pycharmm(),
    reason="pycharmm not available in this environment",
)
def test_mmml_calculator_general_charmm_flag():
    """General MMML calculator module exposes _HAVE_PYCHARMM."""
    from mmml.pycharmmInterface import mmml_calculator_general

    assert hasattr(mmml_calculator_general, "_HAVE_PYCHARMM")
    assert mmml_calculator_general._HAVE_PYCHARMM is True
