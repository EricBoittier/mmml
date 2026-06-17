"""JAX-MD integrator dtype boundary (float32 carry vs ml_compute_dtype float64)."""

import jax.numpy as jnp

from mmml.cli.run.jaxmd_runner import as_jaxmd_dtype, normalize_jaxmd_state


def test_as_jaxmd_dtype_downcasts_float64_forces():
    f64 = jnp.ones((10, 3), dtype=jnp.float64)
    out = as_jaxmd_dtype(f64)
    assert out.dtype == jnp.float32


def test_as_jaxmd_dtype_preserves_float32():
    f32 = jnp.ones((10, 3), dtype=jnp.float32)
    out = as_jaxmd_dtype(f32)
    assert out.dtype == jnp.float32


def test_normalize_jaxmd_state_casts_carry_fields():
    class _State:
        def __init__(self):
            self.position = jnp.ones((2, 3), dtype=jnp.float64)
            self.momentum = jnp.ones((2, 3), dtype=jnp.float64)
            self.mass = jnp.ones((2,), dtype=jnp.float64)

        def set(self, **kwargs):
            out = _State()
            out.position = kwargs.get("position", self.position)
            out.momentum = kwargs.get("momentum", self.momentum)
            out.mass = kwargs.get("mass", self.mass)
            return out

    normed = normalize_jaxmd_state(_State())
    assert normed.position.dtype == jnp.float32
    assert normed.momentum.dtype == jnp.float32
    assert normed.mass.dtype == jnp.float32
