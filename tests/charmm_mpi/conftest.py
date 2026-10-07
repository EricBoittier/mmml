"""Fixtures for live CHARMM MPI tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tip3_charmm_ff(pycharmm_workdir: Path):
    """Load TIP3 water PSF/coords with CGENFF MM terms only (no MLpot)."""
    from karml.interfaces.pycharmmInterface import setupRes
    from karml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from karml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds
    from karml.interfaces.pycharmmInterface.import_pycharmm import (
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
