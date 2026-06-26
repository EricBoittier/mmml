"""Fixtures for CHARMM-only functionality tests (no MLpot)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["tests.functionality.pycharmmETC.conftest"]


@pytest.fixture
def tip3_charmm_ff(pycharmm_workdir: Path):
    """Load TIP3 water PSF/coords with CGENFF MM terms only (no MLpot)."""
    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )

    atoms = setupRes.main("TIP3")
    atoms = setupRes.generate_coordinates()
    reset_block()
    reset_block_no_internal()
    reset_block()
    apply_charmm_mm_block()
    setup_default_nbonds()
    yield atoms
    reset_block()
    reset_block_no_internal()
    reset_block()
