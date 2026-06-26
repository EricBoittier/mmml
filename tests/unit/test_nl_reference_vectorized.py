"""CPU parity tests for dynamic MM neighbor-list helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend
from mmml.interfaces.pycharmmInterface.nl_reference import (
    compare_pair_sets,
    extract_valid_pairs,
    filter_vesin_half_list_vectorized,
    have_vesin,
    monomer_id_from_offsets,
    reference_mic_pairs,
    vesin_mic_pairs,
    vesin_raw_half_list,
)


def _toy_cluster() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.4, 1.0, 1.0],
            [4.0, 1.0, 1.0],
            [4.4, 1.0, 1.0],
            [8.0, 5.0, 3.0],
            [8.4, 5.0, 3.0],
        ],
        dtype=np.float64,
    )
    cell = np.diag([10.0, 12.0, 14.0]).astype(np.float64)
    offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    monomer_id = monomer_id_from_offsets(offsets, positions.shape[0])
    return positions, cell, offsets, monomer_id


@pytest.mark.unit
def test_vectorized_vesin_filter_matches_oracle_with_mm_r_min() -> None:
    if not have_vesin():
        pytest.skip("vesin not installed")

    positions, cell, offsets, monomer_id = _toy_cluster()
    cutoff = 6.0
    mm_r_min = 2.0

    ref = vesin_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )
    i_raw, j_raw, dist_raw = vesin_raw_half_list(positions, cell, cutoff)
    i_vec, j_vec = filter_vesin_half_list_vectorized(
        i_raw,
        j_raw,
        dist_raw,
        cutoff,
        monomer_id,
        positions,
        cell,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )

    got = {(int(i), int(j)) for i, j in zip(i_vec, j_vec, strict=False)}
    assert compare_pair_sets(ref, got).match


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["cell_list", "vesin"])
def test_rebuild_backends_match_reference_pair_set(backend: str) -> None:
    if backend == "vesin" and not have_vesin():
        pytest.skip("vesin not installed")

    positions, cell, offsets, monomer_id = _toy_cluster()
    cutoff = 6.0
    mm_r_min = 2.0
    ref, _source = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
        prefer_vesin=False,
    )

    pair_i, pair_j, mask, _n_valid, _capacity, _used = build_mm_pairs_with_backend(
        backend,
        positions,
        cell,
        cutoff=cutoff,
        monomer_offsets=offsets,
        atoms_per_monomer_list=[2, 2, 2],
        mm_r_min=mm_r_min,
        max_pairs=64,
        total_atoms=positions.shape[0],
    )
    got = extract_valid_pairs(pair_i, pair_j, mask)

    assert compare_pair_sets(ref, got).match


@pytest.mark.unit
def test_dense_liquid_vectorized_filter_matches_oracle() -> None:
    if not have_vesin():
        pytest.skip("vesin not installed")

    from tests.functionality.neighbor_lists._common import (
        build_liquid_density_synthetic_case,
        liquid_density_synthetic_cases,
    )

    cases = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    positions, cell, offsets, monomer_id, cutoff, *_ = build_liquid_density_synthetic_case(
        cases["synthetic_aco_liquid_n32_rho150"]
    )
    i_raw, j_raw, dist_raw = vesin_raw_half_list(positions, cell, cutoff)
    i_vec, j_vec = filter_vesin_half_list_vectorized(
        i_raw,
        j_raw,
        dist_raw,
        cutoff,
        monomer_id,
        positions,
        cell,
        monomer_offsets=offsets,
    )
    got = {(int(i), int(j)) for i, j in zip(i_vec, j_vec, strict=False)}
    ref = vesin_mic_pairs(positions, cell, cutoff, monomer_id, monomer_offsets=offsets)

    assert compare_pair_sets(ref, got).match
