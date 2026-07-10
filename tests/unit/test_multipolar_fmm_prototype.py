"""Executable sketches for a future multipolar FMM backend."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.multipolar_fmm_prototype import (
    CartesianMultipoleLayout,
    direct_multipole_to_multipole_energy,
    direct_multipole_to_point,
    multipole_coeff_count,
    pack_cartesian_multipoles,
    pair_potential_from_cartesian_multipole,
    self_energy,
    unpack_cartesian_multipoles,
)


def test_cartesian_layout_matches_incremental_orders():
    assert multipole_coeff_count(0) == 1
    assert multipole_coeff_count(1) == 4
    assert multipole_coeff_count(2) == 10
    layout = CartesianMultipoleLayout(order=2)
    assert layout.charge_index == 0
    assert layout.dipole_slice == slice(1, 4)
    assert layout.quadrupole_slice == slice(4, 10)


def test_pack_unpack_cartesian_moments_roundtrip():
    charge = jnp.array([1.25, -0.5])
    dipole = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]])
    quadrupole = jnp.array(
        [
            [[2.0, 0.1, 0.2], [0.1, -1.0, 0.3], [0.2, 0.3, -1.0]],
            [[0.5, -0.4, 0.0], [-0.4, 0.25, 0.7], [0.0, 0.7, -0.75]],
        ]
    )
    coeffs = pack_cartesian_multipoles(
        charge,
        dipole=dipole,
        quadrupole=quadrupole,
        order=2,
    )
    got_charge, got_dipole, got_quadrupole = unpack_cartesian_multipoles(coeffs)
    np.testing.assert_allclose(got_charge, charge)
    np.testing.assert_allclose(got_dipole, dipole)
    np.testing.assert_allclose(got_quadrupole, quadrupole)


def test_point_charge_path_matches_direct_coulomb_sum():
    source_positions = jnp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    )
    target_positions = jnp.array([[1.0, 1.0, 0.0], [4.0, 0.0, 0.0]])
    charges = jnp.array([1.0, -2.0, 0.5])
    coeffs = pack_cartesian_multipoles(charges, order=0)

    got = direct_multipole_to_point(source_positions, coeffs, target_positions)
    displacement = target_positions[:, None, :] - source_positions[None, :, :]
    expected = jnp.sum(
        charges[None, :] / jnp.linalg.norm(displacement, axis=-1),
        axis=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_source_dipole_potential_uses_target_minus_source_displacement():
    displacement = jnp.array([2.0, 0.0, 0.0])
    coeffs = pack_cartesian_multipoles(
        jnp.array(0.0),
        dipole=jnp.array([3.0, 0.0, 0.0]),
        order=1,
    )

    got = pair_potential_from_cartesian_multipole(displacement, coeffs)
    assert got == pytest.approx(3.0 / 4.0)


def test_quadrupole_potential_matches_cartesian_hessian_formula():
    displacement = jnp.array([1.0, 2.0, 2.0])
    quadrupole = jnp.diag(jnp.array([2.0, -1.0, -1.0]))
    coeffs = pack_cartesian_multipoles(
        jnp.array(0.0),
        quadrupole=quadrupole,
        order=2,
    )

    got = pair_potential_from_cartesian_multipole(displacement, coeffs)
    r2 = jnp.dot(displacement, displacement)
    hess_green = (3.0 * jnp.outer(displacement, displacement) - jnp.eye(3) * r2) / (
        r2 ** 2.5
    )
    expected = 0.5 * jnp.sum(quadrupole * hess_green)
    assert got == pytest.approx(float(expected), abs=1e-7)


def test_target_dipole_energy_is_dipole_dot_potential_gradient():
    source_positions = jnp.array([[0.0, 0.0, 0.0]])
    source_coeffs = pack_cartesian_multipoles(jnp.array([2.0]), order=0)
    target_positions = jnp.array([[1.0, 0.0, 0.0]])
    target_coeffs = pack_cartesian_multipoles(
        jnp.array([0.0]),
        dipole=jnp.array([[4.0, 0.0, 0.0]]),
        order=1,
    )

    got = direct_multipole_to_multipole_energy(
        source_positions,
        source_coeffs,
        target_positions,
        target_coeffs,
    )
    assert got[0] == pytest.approx(-8.0)


def test_shared_source_target_energy_masks_self_and_halves_pairs():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coeffs = pack_cartesian_multipoles(jnp.array([1.0, -3.0]), order=0)

    per_target = direct_multipole_to_multipole_energy(
        positions,
        coeffs,
        positions,
        coeffs,
        exclude_self=True,
    )
    total = self_energy(positions, coeffs)
    np.testing.assert_allclose(
        per_target,
        jnp.array([-1.5, -1.5]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert total == pytest.approx(-1.5)


def test_self_energy_is_jax_differentiable_force_reference():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coeffs = pack_cartesian_multipoles(jnp.array([1.0, -3.0]), order=0)

    energy_fn = lambda x: self_energy(x.reshape((2, 3)), coeffs)
    forces = -jax.grad(energy_fn)(positions.reshape(-1)).reshape((2, 3))
    expected = jnp.array([[0.75, 0.0, 0.0], [-0.75, 0.0, 0.0]])
    np.testing.assert_allclose(forces, expected, rtol=1e-6, atol=1e-6)
