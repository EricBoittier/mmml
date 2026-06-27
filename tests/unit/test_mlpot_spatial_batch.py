"""Tests for spatial MPI batch builder and policy."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    lattice_positions_cubic_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    build_all_rank_active_sets,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.batch_builder import (
    build_spatial_batch_indices,
    make_spatial_domain_grid,
    per_rank_physnet_budget,
)
from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
    pin_cuda_for_spatial_mpi,
    spatial_mpi_enabled,
)


def test_spatial_mpi_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    assert spatial_mpi_enabled() is False
    assert spatial_mpi_enabled(True) is True


def test_pin_cuda_uses_local_rank_before_jax(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_cuda_for_spatial_mpi() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_cuda_falls_back_to_world_rank(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_RANK", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_cuda_for_spatial_mpi() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_build_spatial_batch_indices_matches_active_set():
    n_monomers = 32
    atoms_per = 10
    box = 40.0
    pos = lattice_positions_cubic_pbc(n_monomers, atoms_per, box, spacing_A=5.0, seed=3)
    cp = CutoffParameters()
    grid = make_spatial_domain_grid(box, 2, cp)
    batch = build_spatial_batch_indices(pos, n_monomers, atoms_per, grid, rank=0, cutoff_params=cp)
    sets = build_all_rank_active_sets(pos, n_monomers, atoms_per, grid, mm_switch_on=cp.mm_switch_on)
    assert np.array_equal(batch.owned_monomers, sets[0].owned_monomers)
    assert np.array_equal(batch.active_dimer_indices, sets[0].active_dimer_indices)
    assert batch.physnet_systems == batch.n_owned_monomers + batch.n_active_dimers


def test_per_rank_physnet_budget():
    assert per_rank_physnet_budget(1000, 6000, 2) == 7000
