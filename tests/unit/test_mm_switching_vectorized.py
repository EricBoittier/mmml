"""Vectorized MM switching helpers (compile-friendly)."""

from __future__ import annotations

import jax.numpy as jnp


def test_mm_scale_jnp_repeat_matches_concat() -> None:
    """Regression: avoid O(n_dimers) Python concat inside JIT."""
    mm_scale = jnp.array([0.1, 0.9, 0.5, 0.0])
    n_pairs = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    repeated = jnp.repeat(mm_scale, n_pairs)
    manual = jnp.concatenate(
        [
            jnp.full((2,), 0.1),
            jnp.full((0,), 0.9),
            jnp.full((3,), 0.5),
            jnp.full((1,), 0.0),
        ]
    )
    assert repeated.shape == manual.shape
    assert bool(jnp.allclose(repeated, manual))
