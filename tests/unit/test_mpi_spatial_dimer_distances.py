"""Unit tests for vectorized spatial dimer COM distances."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    lattice_positions_cubic_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    build_monomer_dimer_index_arrays,
    dimer_com_distance_numpy,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.active_set import (
    global_near_dimer_mask,
    monomer_pair_com_distances_mic,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import compute_monomer_coms


def _reference_dimer_dists(pos, n_monomers, atoms_per, box_side):
    cell = np.array(
        [[box_side, 0.0, 0.0], [0.0, box_side, 0.0], [0.0, 0.0, box_side]],
        dtype=np.float64,
    )
    _, dimer_idx, dimer_n_a, dimer_n_b, _ = build_monomer_dimer_index_arrays(
        n_monomers, atoms_per
    )
    return np.array(
        [
            dimer_com_distance_numpy(
                pos, dimer_idx[di], int(dimer_n_a[di]), int(dimer_n_b[di]), cell
            )
            for di in range(len(dimer_idx))
        ],
        dtype=np.float64,
    )


def test_monomer_pair_com_distances_matches_loop():
    n_monomers = 100
    atoms_per = 5
    box = 25.0
    pos = lattice_positions_cubic_pbc(n_monomers, atoms_per, box, spacing_A=4.0, seed=7)
    pairs, near = global_near_dimer_mask(
        pos, n_monomers, atoms_per, mm_switch_on=8.0, box_side_A=box
    )
    coms = compute_monomer_coms(pos, n_monomers, atoms_per)
    cell = np.array(
        [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
        dtype=np.float64,
    )
    fast = monomer_pair_com_distances_mic(coms, pairs, cell)
    ref = _reference_dimer_dists(pos, n_monomers, atoms_per, box)
    assert fast.shape == ref.shape
    np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-10)
    assert int(np.sum(near)) == int(np.sum(ref < 8.0))


def test_global_near_dimer_mask_dcm100_count():
    n_monomers = 100
    atoms_per = 5
    box = 25.0
    pos = lattice_positions_cubic_pbc(n_monomers, atoms_per, box, spacing_A=4.0, seed=3)
    pairs, near = global_near_dimer_mask(
        pos, n_monomers, atoms_per, mm_switch_on=8.0, box_side_A=box
    )
    assert pairs.shape[0] == n_monomers * (n_monomers - 1) // 2
    assert near.dtype == bool
    assert 0 < int(np.sum(near)) < pairs.shape[0]
