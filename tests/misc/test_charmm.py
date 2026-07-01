"""
Tests for CHARMM / PyCHARMM integration.

CHARMM is optional and requires CHARMM_HOME, CHARMM_LIB_DIR, and the pycharmm
package. These tests verify the interface works when available.
"""

from __future__ import annotations

import os
import pytest


from tests.conftest import can_import_pycharmm, charmm_env_configured


@pytest.mark.skipif(
    not charmm_env_configured(),
    reason="CHARMM_HOME or CHARMM_LIB_DIR not set or paths do not exist",
)
def test_charmm_import():
    """PyCHARMM can be imported when CHARMM env is configured."""
    import pycharmm  # noqa: F401
    assert pycharmm is not None


@pytest.mark.skipif(
    not can_import_pycharmm(),
    reason="pycharmm not available in this environment",
)
def test_mmml_calculator_charmm_flag():
    """MMML calculator module exposes _HAVE_PYCHARMM correctly."""
    from mmml.interfaces.pycharmmInterface import mmml_calculator

    assert hasattr(mmml_calculator, "_HAVE_PYCHARMM")
    assert mmml_calculator._HAVE_PYCHARMM is True

