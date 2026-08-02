"""Unit tests for the linear-combination-of-distances collective variable."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.restraints import LinearDistanceCV, harmonic_bias_energy


# Cl(0) --- C(1) --- N(2) collinear along x: r(C-Cl)=1.8, r(C-N)=3.0
_SN2 = np.array(
    [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [4.8, 0.0, 0.0]],
    dtype=np.float64,
)
_CV = LinearDistanceCV.difference(minuend=(1, 0), subtrahend=(1, 2))


def test_difference_cv_value_matches_hand_computation():
    assert _CV.value_numpy(_SN2) == pytest.approx(1.8 - 3.0)
    assert float(_CV.value(_SN2)) == pytest.approx(1.8 - 3.0)


def test_distance_cv_is_the_degenerate_case():
    cv = LinearDistanceCV.distance(0, 1)
    assert cv.is_plain_distance
    assert cv.value_numpy(_SN2) == pytest.approx(1.8)
    assert float(cv.value(_SN2)) == pytest.approx(1.8)


def test_reactant_negative_product_positive():
    """Turan sign convention: reactants at xi < 0, products at xi > 0."""
    reactant = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [5.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [3.2, 0.0, 0.0], [4.7, 0.0, 0.0]])
    assert _CV.value_numpy(reactant) < 0.0
    assert _CV.value_numpy(product) > 0.0


def test_value_batched_matches_per_frame_value():
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    frames = _SN2[None] + 0.4 * rng.standard_normal((5, 3, 3))
    packed = jnp.asarray(frames.reshape(-1, 3))
    batched = np.asarray(_CV.value_batched(packed, n_atoms=3, n_windows=5))
    per_frame = np.array([_CV.value_numpy(f) for f in frames])
    np.testing.assert_allclose(batched, per_frame, rtol=0, atol=1e-9)


def test_values_numpy_matches_value_numpy_over_trajectory():
    rng = np.random.default_rng(1)
    traj = _SN2[None] + 0.3 * rng.standard_normal((7, 3, 3))
    np.testing.assert_allclose(
        _CV.values_numpy(traj),
        [_CV.value_numpy(f) for f in traj],
        rtol=0,
        atol=1e-12,
    )


def test_analytic_gradient_matches_autodiff():
    """The packed sampler uses gradient_batched instead of AD; they must agree."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2)
    frames = _SN2[None] + 0.5 * rng.standard_normal((4, 3, 3))
    packed = jnp.asarray(frames.reshape(-1, 3))

    analytic = np.asarray(_CV.gradient_batched(packed, n_atoms=3, n_windows=4))
    autodiff = np.asarray(
        jax.grad(lambda r: jnp.sum(_CV.value_batched(r, 3, 4)))(packed)
    ).reshape(4, 3, 3)
    np.testing.assert_allclose(analytic, autodiff, rtol=1e-9, atol=1e-9)


def test_gradient_accumulates_on_the_shared_atom():
    """The carbon appears in both distances; its gradient must be the sum."""
    import jax.numpy as jnp

    grad = np.asarray(
        _CV.gradient_batched(jnp.asarray(_SN2), n_atoms=3, n_windows=1)
    )[0]
    # Collinear: d(xi)/dCl = -(+x), d(xi)/dN = +(+x), d(xi)/dC = +x - x ... check sum
    np.testing.assert_allclose(grad.sum(axis=0), np.zeros(3), atol=1e-9)
    assert not np.allclose(grad[1], 0.0)


def test_harmonic_bias_energy_is_zero_at_target():
    xi = _CV.value_numpy(_SN2)
    assert float(harmonic_bias_energy(xi, target=xi, k=150.0)) == pytest.approx(0.0)
    assert float(harmonic_bias_energy(xi, target=xi + 1.0, k=4.0)) == pytest.approx(2.0)


def test_minimum_image_is_applied_when_a_cell_is_given():
    cell = np.diag([10.0, 10.0, 10.0])
    wrapped = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
    cv = LinearDistanceCV.distance(0, 1)
    assert cv.value_numpy(wrapped) == pytest.approx(9.0)
    assert cv.value_numpy(wrapped, cell=cell) == pytest.approx(1.0)


def test_from_spec_accepts_pairs_instances_and_mappings():
    assert LinearDistanceCV.from_spec(_CV) is _CV
    assert LinearDistanceCV.from_spec((0, 1)) == LinearDistanceCV.distance(0, 1)
    spec = {"pairs": [[1, 0], [1, 2]], "coefficients": [1.0, -1.0]}
    assert LinearDistanceCV.from_spec(spec) == _CV


def test_label_and_atom_indices():
    assert _CV.label() == "r(1-0) - r(1-2)"
    assert LinearDistanceCV.distance(0, 1).label() == "r(0-1)"
    assert (
        LinearDistanceCV(pairs=((0, 1), (1, 2)), coefficients=(0.5, -2.0)).label()
        == "0.5*r(0-1) - 2*r(1-2)"
    )
    assert _CV.atom_indices == (0, 1, 2)
    assert _CV.max_atom_index == 2


@pytest.mark.parametrize(
    ("pairs", "coeffs", "match"),
    [
        ((), (), "at least one atom pair"),
        ((((0, 1)),), (1.0, -1.0), "same length"),
        (((0, 0),), (1.0,), "distinct"),
        (((0, 1),), (0.0,), "non-zero coefficient"),
    ],
)
def test_validation_rejects_malformed_cvs(pairs, coeffs, match):
    with pytest.raises(ValueError, match=match):
        LinearDistanceCV(pairs=pairs, coefficients=coeffs)


def test_validate_against_catches_out_of_range_indices():
    with pytest.raises(ValueError, match="atom index 2"):
        _CV.validate_against(2)
    _CV.validate_against(3)
