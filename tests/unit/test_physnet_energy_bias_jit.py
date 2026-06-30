"""PhysNet use_energy_bias must JIT when checkpoint params are numpy (JSON load)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_energy_bias_numpy_params_index_under_lax_map():
    """Reproduce warmup failure: numpy bias + traced Z inside jax.lax.map."""
    energy_bias = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    z_flat = jnp.array([6, 1, 1, 17, 1] * 256, dtype=jnp.int64)  # 1280, like DCM batch
    n_chunks = 10
    chunk = z_flat.size // n_chunks
    z_chunks = z_flat.reshape(n_chunks, chunk)

    @jax.jit
    def run_take(zc):
        def one_chunk(z):
            bias = jnp.take(jnp.asarray(energy_bias), z)
            return jnp.sum(bias)

        return jax.lax.map(one_chunk, zc)

    out = run_take(z_chunks)
    assert out.shape == (n_chunks,)
    assert jnp.all(jnp.isfinite(out))


def test_energy_bias_numpy_fancy_index_fails_under_lax_map():
    energy_bias = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    z_flat = jnp.array([6, 1, 1, 17, 1] * 256, dtype=jnp.int64)
    n_chunks = 10
    chunk = z_flat.size // n_chunks
    z_chunks = z_flat.reshape(n_chunks, chunk)

    @jax.jit
    def run_fancy(zc):
        def one_chunk(z):
            return jnp.sum(energy_bias[z])

        return jax.lax.map(one_chunk, zc)

    with pytest.raises((jax.errors.TracerArrayConversionError, TypeError, ValueError)):
        run_fancy(z_chunks).block_until_ready()
