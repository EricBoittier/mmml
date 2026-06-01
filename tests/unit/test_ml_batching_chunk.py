"""Tests for chunked ML batch index expansion."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

e3x = pytest.importorskip("e3x")

from mmml.interfaces.pycharmmInterface.ml_batching import prepare_batches_md


def test_prepare_batches_md_chunk_size_smaller_than_data():
    """Regression: e3x 1D pair indices must expand for batch_size < len(R)."""
    num_atoms = 10
    batch_size = 64
    n_systems = batch_size
    data = {
        "R": jnp.zeros((n_systems, num_atoms, 3)),
        "Z": jnp.ones((n_systems, num_atoms), dtype=jnp.int32),
        "N": jnp.full((n_systems,), num_atoms, dtype=jnp.int32),
    }
    batch = prepare_batches_md(data, batch_size=batch_size, num_atoms=num_atoms)[0]
    n_pairs_per_system = e3x.ops.sparse_pairwise_indices(num_atoms)[0].shape[0]
    assert batch["dst_idx"].shape[0] == batch_size * n_pairs_per_system
    assert batch["batch_segments"].shape[0] == batch_size * num_atoms
