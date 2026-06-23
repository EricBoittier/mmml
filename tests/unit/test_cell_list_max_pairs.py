"""Tests for PBC cell-list max_pairs estimation."""

from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.cell_list import (
    cubic_box_side_from_cell_matrix,
    estimate_max_pairs,
)


def test_estimate_max_pairs_dcm_77_cluster() -> None:
    # dcm_77_t50_l32: 385 atoms, L=32 Å, cutoff 10 Å.
    est = estimate_max_pairs(
        385,
        cutoff=10.0,
        safety_factor=3.0,
        box_side_A=32.0,
    )
    assert est >= 91_438


def test_cell_list_pairs_raises_when_buffer_too_small() -> None:
    import numpy as np

    from mmml.interfaces.pycharmmInterface.cell_list import (
        PairListTruncationError,
        cell_list_pairs,
    )

    n = 20
    L = 10.0
    pos = np.random.default_rng(0).uniform(-2, 2, size=(n, 3))
    cell = np.diag([L, L, L])
    offsets = np.arange(0, n + 1, 5, dtype=int)
    with pytest.raises(PairListTruncationError) as exc:
        cell_list_pairs(
            pos,
            cell,
            cutoff=8.0,
            max_pairs=10,
            monomer_offsets=offsets,
            exclude_intra_monomer=True,
        )
    assert exc.value.n_found > 10
    assert exc.value.suggested_max_pairs > 10


def test_estimate_max_pairs_dense_dcm_155_cluster() -> None:
    # dcm_155_t200_l28: 775 atoms, L=28 Å, mm cutoff 10 Å (6+4 from campaign).
    est = estimate_max_pairs(
        775,
        cutoff=10.0,
        safety_factor=1.25,
        box_side_A=28.0,
    )
    assert est >= 372_969


def test_estimate_max_pairs_old_heuristic_underestimated_dense_cluster() -> None:
    bulk_only = estimate_max_pairs(775, cutoff=10.0, safety_factor=3.0)
    assert bulk_only < 298_375


def test_cubic_box_side_from_scalar_and_matrix() -> None:
    assert cubic_box_side_from_cell_matrix(28.0) == 28.0
    import numpy as np

    m = np.diag([28.0, 28.0, 28.0])
    assert cubic_box_side_from_cell_matrix(m) == 28.0


def test_build_cell_list_pairs_with_retry_autoscales() -> None:
    import numpy as np
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import _build_cell_list_pairs_with_retry

    n = 20
    L = 10.0
    pos = np.random.default_rng(0).uniform(-2, 2, size=(n, 3))
    cell = np.diag([L, L, L])
    offsets = np.arange(0, n + 1, 5, dtype=int)

    cl_i, cl_j, cl_mask, n_valid, capacity = _build_cell_list_pairs_with_retry(
        positions=pos,
        pbc_cell=cell,
        cutoff=8.0,
        max_pairs=2,
        monomer_offsets=offsets,
        atoms_per_monomer_list=[5] * 4,
        total_atoms=n,
        cell_list_safety_factor=1.5,
        cell_list_density_estimate=None,
        debug=False,
    )
    assert capacity > 2
    assert len(cl_i) == capacity
    assert n_valid > 2
    assert cl_mask[:n_valid].all()
    assert not cl_mask[n_valid:].any()
