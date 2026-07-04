"""Vectorized MM switching helpers (compile-friendly)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


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


def test_mean_switch_scale_jax_matches_numpy_reference() -> None:
    """Vectorized JAX COM switch must match the numpy reference implementation."""
    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        _mean_switch_scale,
        _mean_switch_scale_jax,
    )

    rng = np.random.default_rng(0)
    n_monomers = 6
    atoms_per = 5
    n_atoms = n_monomers * atoms_per
    offsets = np.arange(0, n_atoms + 1, atoms_per, dtype=np.int64)
    pos = rng.normal(size=(n_atoms, 3))
    cell = np.diag([28.0, 28.0, 28.0])
    kwargs = dict(
        ml_switch_width=1.0,
        mm_switch_on=8.0,
        mm_switch_width=2.0,
        complementary_handoff=True,
        mm_r_min=None,
    )
    ref = _mean_switch_scale(
        pos,
        offsets,
        pbc_cell=cell,
        **kwargs,
    )
    jax_val = float(
        _mean_switch_scale_jax(
            jnp.asarray(pos, dtype=jnp.float64),
            offsets,
            jnp.asarray(cell, dtype=jnp.float64),
            **kwargs,
        )
    )
    assert jax_val == pytest.approx(ref, rel=0, abs=1e-10)
