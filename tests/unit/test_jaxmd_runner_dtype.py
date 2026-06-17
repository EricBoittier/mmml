"""JAX-MD integrator dtype boundary (float32 carry vs ml_compute_dtype float64)."""

import jax.numpy as jnp

from mmml.cli.run.jaxmd_runner import as_jaxmd_dtype


def test_as_jaxmd_dtype_downcasts_float64_forces():
    f64 = jnp.ones((10, 3), dtype=jnp.float64)
    out = as_jaxmd_dtype(f64)
    assert out.dtype == jnp.float32


def test_as_jaxmd_dtype_preserves_float32():
    f32 = jnp.ones((10, 3), dtype=jnp.float32)
    out = as_jaxmd_dtype(f32)
    assert out.dtype == jnp.float32
