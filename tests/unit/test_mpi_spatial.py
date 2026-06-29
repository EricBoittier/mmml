"""Unit tests for spatial MPI ML decomposition (mocked geometry, no CHARMM)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    lattice_positions_cubic_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    build_all_rank_active_sets,
    global_near_dimer_mask,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.dedup import (
    assign_canonical_dimer_owner,
    union_active_dimer_ids,
    verify_unique_dimer_coverage,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    SpatialDomainGrid,
    compute_monomer_coms,
    resolve_halo_radius,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    build_monomer_dimer_index_arrays,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_info import survey_domdec_api
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange import (
    merge_partial_forces,
)


def _fixture_geometry(
    n_monomers: int = 64,
    box: float = 40.0,
    spacing: float = 5.0,
    seed: int = 7,
):
    atoms_per = 10
    pos = lattice_positions_cubic_pbc(
        n_monomers, atoms_per, box, spacing_A=spacing, seed=seed
    )
    return pos, n_monomers, atoms_per, box


def test_resolve_halo_radius_default():
    r = resolve_halo_radius()
    assert r > 14.0
    assert r < 20.0


def test_survey_domdec_api_capabilities():
    survey = survey_domdec_api()
    assert survey.charmm_fortran_domdec is True
    # Both atom-map APIs are now implemented via domdec_atoms.py ctypes reader
    assert survey.pycharmm_local_atom_api is True
    assert survey.pycharmm_ghost_atom_api is True
    assert survey.mmml_disable_domdec_for_mlpot is True


def test_rank_for_com_slab_partition():
    grid = SpatialDomainGrid(box_side_A=40.0, n_ranks=4, halo_radius_A=10.0)
    coms = compute_monomer_coms(
        lattice_positions_cubic_pbc(32, 10, 40.0, spacing_A=5.0),
        32,
        10,
    )
    ranks = [grid.rank_for_com(c) for c in coms]
    assert min(ranks) >= 0
    assert max(ranks) < 4


def test_halo_extends_visibility():
    pos, n_monomers, atoms_per, box = _fixture_geometry()
    halo = resolve_halo_radius()
    grid = SpatialDomainGrid(box_side_A=box, n_ranks=4, halo_radius_A=halo)
    coms = compute_monomer_coms(pos, n_monomers, atoms_per)
    owned_counts = [int(np.sum(grid.owned_monomer_mask(coms, r))) for r in range(4)]
    extended_counts = [
        int(np.sum(grid.monomers_in_extended_domain(coms, r))) for r in range(4)
    ]
    assert sum(owned_counts) == n_monomers
    assert all(e >= o for e, o in zip(extended_counts, owned_counts))
    assert sum(extended_counts) > sum(owned_counts)


def test_deduplicated_dimer_union_covers_global_near():
    pos, n_monomers, atoms_per, box = _fixture_geometry()
    halo = resolve_halo_radius()
    grid = SpatialDomainGrid(box_side_A=box, n_ranks=4, halo_radius_A=halo)
    active_sets = build_all_rank_active_sets(
        pos, n_monomers, atoms_per, grid, mm_switch_on=8.0
    )
    pairs, near = global_near_dimer_mask(pos, n_monomers, atoms_per, box_side_A=box)
    near_ids = np.nonzero(near)[0].astype(np.int32)
    assert verify_unique_dimer_coverage(active_sets, near_ids)

    union = union_active_dimer_ids(active_sets)
    assert union.shape == near_ids.shape


def test_each_near_dimer_owned_by_exactly_one_rank():
    pos, n_monomers, atoms_per, box = _fixture_geometry()
    halo = resolve_halo_radius()
    grid = SpatialDomainGrid(box_side_A=box, n_ranks=4, halo_radius_A=halo)
    active_sets = build_all_rank_active_sets(
        pos, n_monomers, atoms_per, grid, mm_switch_on=8.0
    )
    pairs, near = global_near_dimer_mask(pos, n_monomers, atoms_per, box_side_A=box)
    coms = compute_monomer_coms(pos, n_monomers, atoms_per)
    cell = np.diag([box, box, box])
    for di in np.nonzero(near)[0]:
        i, j = int(pairs[di, 0]), int(pairs[di, 1])
        owner = assign_canonical_dimer_owner(i, j, coms[i], coms[j], grid, cell=cell)
        holders = [r for r, s in enumerate(active_sets) if di in s.active_dimer_indices]
        assert holders == [owner]


def test_force_conservation_vs_global_reference():
    """Per-rank unique dimer partial forces sum to the global reference."""
    pos, n_monomers, atoms_per, box = _fixture_geometry(n_monomers=32)
    halo = resolve_halo_radius()
    grid = SpatialDomainGrid(box_side_A=box, n_ranks=2, halo_radius_A=halo)
    active_sets = build_all_rank_active_sets(
        pos, n_monomers, atoms_per, grid, mm_switch_on=8.0
    )
    pairs, near = global_near_dimer_mask(pos, n_monomers, atoms_per, box_side_A=box)
    _, dimer_idx, dimer_n_a, dimer_n_b, _ = build_monomer_dimer_index_arrays(
        n_monomers, atoms_per
    )
    n_atoms = n_monomers * atoms_per
    def dimer_force_contribution(di: int) -> np.ndarray:
        r = np.random.default_rng(int(di) + 99)
        f = np.zeros((n_atoms, 3), dtype=np.float64)
        idxs = dimer_idx[di, : int(dimer_n_a[di]) + int(dimer_n_b[di])]
        f[idxs] += r.normal(size=(len(idxs), 3))
        return f

    global_f = np.zeros((n_atoms, 3), dtype=np.float64)
    for di in np.nonzero(near)[0]:
        global_f += dimer_force_contribution(int(di))

    partials = []
    for s in active_sets:
        pf = np.zeros((n_atoms, 3), dtype=np.float64)
        for di in s.active_dimer_indices:
            pf += dimer_force_contribution(int(di))
        partials.append(pf)

    merged = merge_partial_forces(partials, n_atoms=n_atoms)
    np.testing.assert_allclose(merged, global_f, rtol=0, atol=1e-12)
