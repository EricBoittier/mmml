"""Tests for medium PBC sparse dimer cap validation (500–2000 monomers)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    MEDIUM_PBC_MONOMER_COUNTS,
    lattice_positions_cubic_pbc,
    suggest_medium_pbc_sizing,
    validate_medium_pbc_geometry,
    workflow_checklist,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    resolve_max_active_dimers,
)


@pytest.mark.parametrize("n_monomers", MEDIUM_PBC_MONOMER_COUNTS)
def test_medium_pbc_default_cap_policy(n_monomers: int):
    cap = resolve_max_active_dimers(n_monomers, free_space=False)
    assert cap == max(1000, 6 * n_monomers)


@pytest.mark.parametrize("n_monomers", MEDIUM_PBC_MONOMER_COUNTS)
def test_suggest_medium_pbc_sizing_bounds(n_monomers: int):
    sizing = suggest_medium_pbc_sizing(n_monomers)
    assert sizing.n_monomers == n_monomers
    assert sizing.max_active_dimers_cap == max(1000, 6 * n_monomers)
    assert sizing.physnet_systems_upper_bound == n_monomers + sizing.max_active_dimers_cap
    assert sizing.ml_batch_size_gpu == 256
    assert sizing.expected_gpu_chunks is not None
    assert sizing.expected_gpu_chunks >= 1


def test_validate_medium_pbc_sparse_lattice_ok():
    """Wide box + moderate spacing keeps near-dimer count below default cap."""
    n_monomers = 500
    atoms_per = 10
    box = 80.0
    spacing = 6.0
    pos = lattice_positions_cubic_pbc(
        n_monomers, atoms_per, box, spacing_A=spacing, seed=42
    )
    stats = validate_medium_pbc_geometry(
        pos, n_monomers, atoms_per, box_side_A=box
    )
    assert stats["ok"] is True
    assert not stats["cap_saturated"]
    assert stats["n_near_mm_switch_on"] < stats["max_active_dimers_cap"]


def test_validate_medium_pbc_cap_saturated_detected():
    """Dense lattice in a small box exceeds a low explicit cap."""
    n_monomers = 200
    atoms_per = 10
    box = 25.0
    spacing = 3.0
    pos = lattice_positions_cubic_pbc(
        n_monomers, atoms_per, box, spacing_A=spacing, seed=1
    )
    stats = validate_medium_pbc_geometry(
        pos,
        n_monomers,
        atoms_per,
        box_side_A=box,
        max_active_dimers=50,
    )
    assert stats["ok"] is False
    assert stats["cap_saturated"]


def test_workflow_checklist_nonempty():
    lines = workflow_checklist(1000)
    assert len(lines) >= 4
    assert any("validate_mlpot_sparse_dimers" in line for line in lines)
