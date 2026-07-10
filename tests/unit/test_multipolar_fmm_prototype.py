"""Executable sketches for a future multipolar FMM backend."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.multipolar_fmm_prototype import (
    CartesianMultipoleLayout,
    E3X_CONVENTION_NOTE,
    SR_ML_FMM_COMPOSITION_NOTE,
    direct_multipole_to_multipole_energy,
    direct_multipole_to_point,
    mm_mm_multipolar_energy,
    mm_to_ml_multipolar_embedding,
    multipole_coeff_count,
    pack_cartesian_multipoles,
    pair_potential_from_cartesian_multipole,
    self_energy,
    symmetric_traceless,
    unpack_cartesian_multipoles,
)


def test_design_notes_cover_e3x_and_short_range_ml_ownership():
    assert "e3x.Config" in E3X_CONVENTION_NOTE
    assert "symmetric traceless degree-2" in E3X_CONVENTION_NOTE
    assert "do not add a full explicit FMM energy" in SR_ML_FMM_COMPOSITION_NOTE
    assert "short-range residual" in SR_ML_FMM_COMPOSITION_NOTE
    assert "smooth long-range FMM complement" in SR_ML_FMM_COMPOSITION_NOTE


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


def test_quadrupole_pack_projects_to_e3x_traceless_symmetric_default():
    quadrupole = jnp.array(
        [
            [2.0, 3.0, 0.0],
            [1.0, 4.0, 5.0],
            [0.0, 7.0, 9.0],
        ]
    )
    coeffs = pack_cartesian_multipoles(
        jnp.array(0.0),
        quadrupole=quadrupole,
        order=2,
    )

    _, _, got_quadrupole = unpack_cartesian_multipoles(coeffs)
    expected = symmetric_traceless(quadrupole)
    np.testing.assert_allclose(got_quadrupole, expected, rtol=1e-6, atol=1e-6)
    assert jnp.trace(got_quadrupole) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(got_quadrupole, got_quadrupole.T, rtol=1e-6, atol=1e-6)


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


def test_quadrupole_trace_can_be_preserved_for_legacy_cartesian_oracles():
    quadrupole = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    coeffs = pack_cartesian_multipoles(
        jnp.array(0.0),
        quadrupole=quadrupole,
        order=2,
        quadrupole_trace_policy="preserve",
    )

    _, _, got_quadrupole = unpack_cartesian_multipoles(coeffs)
    np.testing.assert_allclose(got_quadrupole, quadrupole)


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


def test_mm_mm_wrapper_owns_pairs_once():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coeffs = pack_cartesian_multipoles(jnp.array([1.0, -3.0]), order=0)

    result = mm_mm_multipolar_energy(positions, coeffs)
    np.testing.assert_allclose(
        result.per_site_energy,
        jnp.array([-1.5, -1.5]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.energy == pytest.approx(-1.5)


def test_mm_to_ml_embedding_is_one_way_without_half_factor():
    mm_positions = jnp.array([[0.0, 0.0, 0.0]])
    ml_positions = jnp.array([[2.0, 0.0, 0.0]])
    mm_coeffs = pack_cartesian_multipoles(jnp.array([2.0]), order=0)
    ml_coeffs = pack_cartesian_multipoles(jnp.array([3.0]), order=0)

    result = mm_to_ml_multipolar_embedding(
        mm_positions,
        mm_coeffs,
        ml_positions,
        ml_target_coeffs=ml_coeffs,
    )
    np.testing.assert_allclose(result.potential, jnp.array([1.0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result.potential_gradient,
        jnp.array([[-0.5, 0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.electric_field,
        jnp.array([[0.5, 0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.target_energy is not None
    np.testing.assert_allclose(result.target_energy, jnp.array([3.0]), rtol=1e-6, atol=1e-6)


def test_self_energy_is_jax_differentiable_force_reference():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coeffs = pack_cartesian_multipoles(jnp.array([1.0, -3.0]), order=0)

    energy_fn = lambda x: self_energy(x.reshape((2, 3)), coeffs)
    forces = -jax.grad(energy_fn)(positions.reshape(-1)).reshape((2, 3))
    expected = jnp.array([[0.75, 0.0, 0.0], [-0.75, 0.0, 0.0]])
    np.testing.assert_allclose(forces, expected, rtol=1e-6, atol=1e-6)
