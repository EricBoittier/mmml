"""Unit tests for Vesin GPU / vectorized NL helpers (skip without GPU when needed)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.nl_reference import (
    filter_vesin_half_list_vectorized,
    have_vesin,
    monomer_id_from_offsets,
    vesin_mic_pairs,
    vesin_raw_half_list,
)


@pytest.mark.skipif(not have_vesin(), reason="vesin not installed")
def test_filter_vesin_half_list_vectorized_matches_set_oracle():
    rng = np.random.default_rng(0)
    n = 24
    L = 20.0
    positions = rng.uniform(0.0, L, size=(n, 3))
    cell = np.diag([L, L, L])
    offsets = np.array([0, 8, 16, n], dtype=np.int32)
    monomer_id = monomer_id_from_offsets(offsets, n)
    cutoff = 8.0

    ref = vesin_mic_pairs(positions, cell, cutoff, monomer_id, monomer_offsets=offsets)
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
    vec_set = {(int(a), int(b)) for a, b in zip(i_vec, j_vec, strict=False)}
    assert vec_set == ref


@pytest.mark.skipif(not have_vesin(), reason="vesin not installed")
def test_gpu_rebuild_parity_when_available():
    pytest.importorskip("cupy")
    import os

    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.nl_gpu import gpu_nl_path_available, rebuild_vesin_pairs_gpu
    from mmml.interfaces.pycharmmInterface.nl_reference import extract_valid_pairs

    if not jax.devices("gpu"):
        pytest.skip("no JAX GPU device")

    prev = os.environ.get("MMML_MM_NL_DEVICE")
    os.environ["MMML_MM_NL_DEVICE"] = "gpu"
    try:
        if not gpu_nl_path_available():
            pytest.skip("cupy or vesin GPU path unavailable")
    finally:
        if prev is None:
            os.environ.pop("MMML_MM_NL_DEVICE", None)
        else:
            os.environ["MMML_MM_NL_DEVICE"] = prev

    rng = np.random.default_rng(1)
    n = 20
    L = 18.0
    positions = rng.uniform(0.0, L, size=(n, 3))
    cell = np.diag([L, L, L])
    offsets = np.array([0, 10, n], dtype=np.int32)
    monomer_id = monomer_id_from_offsets(offsets, n)
    cutoff = 7.0
    ref = vesin_mic_pairs(positions, cell, cutoff, monomer_id, monomer_offsets=offsets)

    os.environ["MMML_MM_NL_DEVICE"] = "gpu"
    device = jax.devices("gpu")[0]
    pos_jax = jax.device_put(jnp.asarray(positions), device)
    pair_idx, pair_mask, _ = rebuild_vesin_pairs_gpu(
        pos_jax,
        cell,
        cutoff=cutoff,
        monomer_offsets=offsets,
        total_atoms=n,
    )
    gpu_set = extract_valid_pairs(
        np.asarray(jax.device_get(pair_idx)),
        np.asarray(jax.device_get(pair_mask), dtype=bool),
    )
    assert gpu_set == ref
