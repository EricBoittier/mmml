"""Unit tests for harmonic flat-bottom COM restraints (``apply_flat_bottom``)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.calculator_utils import (
    FLAT_BOTTOM_MODES,
    apply_flat_bottom,
)


def _mic_identity(a, b, _cell):
    return b - a


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,expected_e,expected_com_dist",
    [
        ("system", 0.0, 0.0),
        ("monomer", 8.0, 5.0),
    ],
)
def test_flat_bottom_system_vs_monomer(mode, expected_e, expected_com_dist) -> None:
    """Two equal-mass monomers at ±5 Å: system COM at origin; each monomer penalized."""
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
    assert float(com_dist) == pytest.approx(expected_com_dist)
    if mode == "monomer":
        assert float(jnp.max(jnp.abs(flat_f))) > 0.0


@pytest.mark.unit
def test_flat_bottom_inside_radius_zero_energy_and_force() -> None:
    positions = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float64)
    atomic_numbers = jnp.array([1], dtype=jnp.int32)
    base_forces = jnp.zeros((1, 3), dtype=jnp.float64)
    monomer_offsets = jnp.array([0, 1], dtype=jnp.int32)

    flat_e, flat_f, com, com_dist = apply_flat_bottom(
        positions,
        atomic_numbers,
        base_forces,
        radius=5.0,
        k=2.0,
        mode="system",
        monomer_offsets=monomer_offsets,
        n_monomers=1,
        pbc_cell=None,
        mic_fn=_mic_identity,
    )

    assert float(flat_e) == 0.0
    assert np.allclose(np.asarray(flat_f), 0.0)
    assert float(com_dist) == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray(com), [1.0, 0.0, 0.0])


@pytest.mark.unit
def test_flat_bottom_harmonic_energy_system_mode() -> None:
    """Single H at 5 Å from origin: E = k (|r| - R)² with R=3, k=1 → E=4."""
    positions = jnp.array([[5.0, 0.0, 0.0]], dtype=jnp.float64)
    atomic_numbers = jnp.array([1], dtype=jnp.int32)
    base_forces = jnp.zeros((1, 3), dtype=jnp.float64)
    monomer_offsets = jnp.array([0, 1], dtype=jnp.int32)

    flat_e, flat_f, _, com_dist = apply_flat_bottom(
        positions,
        atomic_numbers,
        base_forces,
        radius=3.0,
        k=1.0,
        mode="system",
        monomer_offsets=monomer_offsets,
        n_monomers=1,
        pbc_cell=None,
        mic_fn=_mic_identity,
    )

    assert float(com_dist) == pytest.approx(5.0)
    assert float(flat_e) == pytest.approx(4.0)
    # Restoring force on atom: F_x = -k * 2 * excess = -4
    assert float(flat_f[0, 0]) == pytest.approx(-4.0)


@pytest.mark.unit
def test_flat_bottom_forces_match_energy_gradient() -> None:
    """Returned flat_F should be -∂E/∂R for the flat-bottom term alone."""
    positions = jnp.array(
        [[4.0, 0.5, 0.0], [-2.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.array([1, 1, 8], dtype=jnp.int32)
    monomer_offsets = jnp.array([0, 3], dtype=jnp.int32)

    def energy_only(pos):
        e, _, _, _ = apply_flat_bottom(
            pos,
            atomic_numbers,
            jnp.zeros_like(pos),
            radius=2.0,
            k=0.75,
            mode="system",
            monomer_offsets=monomer_offsets,
            n_monomers=1,
            pbc_cell=None,
            mic_fn=_mic_identity,
        )
        return e

    pos = positions
    grad_e = jax.grad(energy_only)(pos)
    flat_e, flat_f, _, _ = apply_flat_bottom(
        pos,
        atomic_numbers,
        jnp.zeros_like(pos),
        radius=2.0,
        k=0.75,
        mode="system",
        monomer_offsets=monomer_offsets,
        n_monomers=1,
        pbc_cell=None,
        mic_fn=_mic_identity,
    )

    assert float(flat_e) > 0.0
    np.testing.assert_allclose(np.asarray(flat_f), -np.asarray(grad_e), rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_flat_bottom_pbc_uses_cell_center() -> None:
    """With an orthorhombic cell, COM is measured relative to the box center via MIC."""
    cell = jnp.diag(jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64))
    # COM at (10, 10, 13): 3 Å above box center (10, 10, 10)
    positions = jnp.array([[10.0, 10.0, 13.0]], dtype=jnp.float64)
    atomic_numbers = jnp.array([1], dtype=jnp.int32)
    monomer_offsets = jnp.array([0, 1], dtype=jnp.int32)

    from mmml.interfaces.pycharmmInterface.calculator_utils import mic_displacement

    flat_e, _, _, com_dist = apply_flat_bottom(
        positions,
        atomic_numbers,
        jnp.zeros((1, 3)),
        radius=2.0,
        k=1.0,
        mode="system",
        monomer_offsets=monomer_offsets,
        n_monomers=1,
        pbc_cell=cell,
        mic_fn=mic_displacement,
    )

    assert float(com_dist) == pytest.approx(3.0)
    assert float(flat_e) == pytest.approx(1.0)  # k * (3 - 2)^2


@pytest.mark.unit
def test_flat_bottom_invalid_mode_raises() -> None:
    positions = jnp.zeros((1, 3))
    atomic_numbers = jnp.array([1])
    monomer_offsets = jnp.array([0, 1])

    with pytest.raises(ValueError, match="flat_bottom mode"):
        apply_flat_bottom(
            positions,
            atomic_numbers,
            jnp.zeros((1, 3)),
            radius=1.0,
            k=1.0,
            mode="cluster",
            monomer_offsets=monomer_offsets,
            n_monomers=1,
            pbc_cell=None,
            mic_fn=_mic_identity,
        )

    assert "cluster" not in FLAT_BOTTOM_MODES
