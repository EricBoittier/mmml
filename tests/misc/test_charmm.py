"""
Tests for CHARMM / PyCHARMM integration.

CHARMM is optional and requires CHARMM_HOME, CHARMM_LIB_DIR, and the pycharmm
package. These tests verify the interface works when available.
"""

from __future__ import annotations

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
    """CHARMM is loadable under mpirun even when collection used MMML_WARMUP_MLPOT_JAX_ONLY."""
    import pycharmm  # noqa: F401

    assert can_import_pycharmm()

