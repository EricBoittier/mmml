"""Tests for PhysNetJax batch preparation."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _mock_data(n: int = 256, natoms: int = 10):
    key = jax.random.PRNGKey(0)
    return {
        "R": jax.random.normal(key, (n, natoms, 3)),
        "F": jax.random.normal(key, (n, natoms, 3)),
        "E": jax.random.normal(key, (n, 1)),
        "Z": jax.random.randint(key, (n, natoms), 1, 10),
        "N": jnp.full((n,), natoms, dtype=jnp.int32),
    }


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_prepare_batches_fast_matches_jit(batch_size: int):
    from mmml.models.physnetjax.physnetjax.data.batches import (
        _pair_indices,
        prepare_batches_fast,
        prepare_batches_jit,
    )

    data = _mock_data(n=256, natoms=10)
    keys = ("R", "Z", "F", "E", "N")
    key = jax.random.PRNGKey(123)
    pair_cache = _pair_indices(10, batch_size)

    ref = prepare_batches_jit(
        key, data, batch_size, data_keys=keys, num_atoms=10
    )
    fast = prepare_batches_fast(
        key,
        data,
        batch_size,
        data_keys=keys,
        num_atoms=10,
        pair_cache=pair_cache,
    )
    assert len(ref) == len(fast)
    for rb, fb in zip(ref, fast):
        for k in ("R", "F", "E", "Z", "N", "dst_idx", "src_idx", "batch_mask", "atom_mask"):
            np.testing.assert_allclose(
                np.asarray(rb[k]),
                np.asarray(fb[k]),
                rtol=1e-6,
                atol=1e-6,
                err_msg=k,
            )


def test_prepare_batches_fast_faster_than_jit():
    from mmml.models.physnetjax.physnetjax.data.batches import (
        _pair_indices,
        prepare_batches_fast,
        prepare_batches_jit,
    )

    data = _mock_data(n=4000, natoms=10)
    keys = ("R", "Z", "F", "E", "N")
    key = jax.random.PRNGKey(7)
    pair_cache = _pair_indices(10, 32)

    _ = prepare_batches_jit(key, data, 32, data_keys=keys, num_atoms=10)
    t0 = time.perf_counter()
    for i in range(5):
        prepare_batches_jit(
            jax.random.fold_in(key, i), data, 32, data_keys=keys, num_atoms=10
        )
    jit_s = time.perf_counter() - t0

    _ = prepare_batches_fast(
        key, data, 32, data_keys=keys, num_atoms=10, pair_cache=pair_cache
    )
    t0 = time.perf_counter()
    for i in range(5):
        prepare_batches_fast(
            jax.random.fold_in(key, i),
            data,
            32,
            data_keys=keys,
            num_atoms=10,
            pair_cache=pair_cache,
        )
    fast_s = time.perf_counter() - t0

    assert fast_s < jit_s
