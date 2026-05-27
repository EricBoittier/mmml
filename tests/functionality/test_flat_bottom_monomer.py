"""Unit tests for per-monomer flat-bottom restraint."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.calculator_utils import apply_flat_bottom


def _mic_identity(a, b, _cell):
    return b - a


@pytest.mark.parametrize("mode,expected_e", [("system", 0.0), ("monomer", 8.0)])
def test_flat_bottom_system_vs_monomer(mode, expected_e):
    """Two equal-mass monomers at ±5 Å: system COM at origin; monomers each penalized."""
    positions = jnp.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], dtype=jnp.float32)
    atomic_numbers = jnp.array([1, 1], dtype=jnp.int32)
    base_forces = jnp.zeros((2, 3), dtype=jnp.float32)
    monomer_offsets = jnp.array([0, 1, 2], dtype=jnp.int32)

    flat_e, flat_f, _com, com_dist = apply_flat_bottom(
        positions,
        atomic_numbers,
        base_forces,
        radius=3.0,
        k=1.0,
        mode=mode,
        monomer_offsets=monomer_offsets,
        n_monomers=2,
        pbc_cell=None,
        mic_fn=_mic_identity,
    )

    assert float(flat_e) == pytest.approx(expected_e)
    if mode == "system":
        assert float(com_dist) == pytest.approx(0.0)
    else:
        assert float(com_dist) == pytest.approx(5.0)
        assert np.allclose(np.asarray(flat_f), 0.0) is False
