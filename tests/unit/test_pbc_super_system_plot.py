"""Unit tests for PBC pedagogy geometry (no matplotlib display)."""

from __future__ import annotations

import numpy as np

from mmml.utils.pbc_super_system_plot import (
    charmm_super_system_atoms,
    four_waters_cubic_cell,
)


def test_four_waters_cell_shape_and_pbc() -> None:
    atoms = four_waters_cubic_cell(side_A=14.0)
    assert len(atoms) == 12
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell.lengths(), [14.0, 14.0, 14.0])


def test_charmm_super_system_replicates_primary_shell() -> None:
    primary = four_waters_cubic_cell(side_A=10.0)
    super_atoms, tags = charmm_super_system_atoms(primary, shell=1)
    n_images = int(np.sum(tags == 1))
    n_primary = int(np.sum(tags == 0))
    assert n_primary == len(primary)
    assert n_images == 26 * len(primary)
