"""Unit tests for LinearDistanceCV / FlatBottomWall (umbrella collective variables)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.restraints import (
    FlatBottomWall,
    LinearDistanceCV,
    harmonic_bias_energy,
    linear_cvs_from_pairs,
)


def _colinear_three() -> np.ndarray:
    # Atoms at 0, 2, 5 Å along x → r(0,1)=2, r(0,2)=5, r(1,2)=3
    return np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )


def test_distance_factory_and_value():
    cv = LinearDistanceCV.distance(0, 1)
    assert cv.pairs == ((0, 1),)
    assert cv.coefficients == (1.0,)
    assert cv.value_numpy(_colinear_three()) == pytest.approx(2.0)
    # Labels are display/metadata only (printed, and stored as "cv_label" in
    # run summaries); pairs render as r(i-j) so multi-term labels stay readable
    # when joined into a list.
    assert cv.label() == "r(0-1)"


def test_difference_factory():
    cv = LinearDistanceCV.difference((0, 1), (0, 2))
    # r01 - r02 = 2 - 5 = -3
    assert cv.value_numpy(_colinear_three()) == pytest.approx(-3.0)
    assert cv.coefficients == (1.0, -1.0)


def test_from_spec_round_trips():
    cv = LinearDistanceCV(pairs=((2, 0), (2, 1)), coefficients=(1.0, 1.0))
    assert LinearDistanceCV.from_spec(cv) is cv
    assert LinearDistanceCV.from_spec((0, 1)).pairs == ((0, 1),)
    rebuilt = LinearDistanceCV.from_spec(
        {"pairs": [[2, 0], [2, 1]], "coefficients": [1.0, -1.0]}
    )
    assert rebuilt.pairs == ((2, 0), (2, 1))
    assert rebuilt.coefficients == (1.0, -1.0)


def test_validate_against_rejects_oob():
    cv = LinearDistanceCV.distance(0, 2)
    cv.validate_against(3)
    with pytest.raises(ValueError, match="references atom index 2 but the system has 2"):
        cv.validate_against(2)


def test_value_and_gradient_batched():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    cv = LinearDistanceCV.difference((0, 1), (0, 2))
    r = _colinear_three()
    packed = np.tile(r[None, :, :], (2, 1, 1)).reshape(6, 3)
    values = np.asarray(cv.value_batched(jnp.asarray(packed), 3, 2))
    np.testing.assert_allclose(values, [-3.0, -3.0])
    grad = np.asarray(cv.gradient_batched(jnp.asarray(packed), 3, 2))
    assert grad.shape == (2, 3, 3)
    # Finite-difference check on window 0, atom 1 x-coordinate
    eps = 1e-5
    r_p = r.copy()
    r_p[1, 0] += eps
    r_m = r.copy()
    r_m[1, 0] -= eps
    fd = (cv.value_numpy(r_p) - cv.value_numpy(r_m)) / (2 * eps)
    assert grad[0, 1, 0] == pytest.approx(fd, abs=1e-5)


def test_flat_bottom_wall_upper_zero_inside():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    wall = FlatBottomWall(
        cv=LinearDistanceCV(pairs=((0, 1), (0, 2)), coefficients=(1.0, 1.0)),
        upper=8.0,
        k=50.0,
    )
    # sum = 2 + 5 = 7 < 8 → zero
    packed = _colinear_three().reshape(3, 3)
    e = float(wall.energy_batched(jnp.asarray(packed), 3, 1)[0])
    assert e == pytest.approx(0.0)
    f = np.asarray(wall.forces_batched(jnp.asarray(packed), 3, 1))
    np.testing.assert_allclose(f, 0.0, atol=1e-12)


def test_flat_bottom_wall_upper_penalizes_outside():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    wall = FlatBottomWall(
        cv=LinearDistanceCV(pairs=((0, 1), (0, 2)), coefficients=(1.0, 1.0)),
        upper=6.0,
        k=2.0,
    )
    # sum = 7 → overshoot 1 → W = 0.5 * 2 * 1^2 = 1
    packed = _colinear_three().reshape(3, 3)
    e = float(wall.energy_batched(jnp.asarray(packed), 3, 1)[0])
    assert e == pytest.approx(1.0)


def test_wall_to_spec_from_spec_round_trip():
    wall = FlatBottomWall(
        cv=LinearDistanceCV.difference((2, 0), (2, 1)),
        upper=6.5,
        k=50.0,
    )
    rebuilt = FlatBottomWall.from_spec(wall.to_spec())
    assert rebuilt.cv.pairs == wall.cv.pairs
    assert rebuilt.cv.coefficients == wall.cv.coefficients
    assert rebuilt.upper == pytest.approx(6.5)
    assert rebuilt.k == pytest.approx(50.0)


def test_harmonic_bias_and_linear_cvs_from_pairs():
    jax = pytest.importorskip("jax")
    e = float(harmonic_bias_energy(jax.numpy.asarray(3.0), 2.0, 4.0))
    assert e == pytest.approx(2.0)  # 0.5*4*(1)^2
    cvs = linear_cvs_from_pairs([(0, 1), (2, 3)])
    assert len(cvs) == 2
    assert cvs[1].pairs == ((2, 3),)


def test_value_numpy_mic_orthorhombic():
    # Atom 1 is +9 along x in a 10 Å box → MIC distance to origin is 1 Å
    pos = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float64)
    cv = LinearDistanceCV.distance(0, 1)
    assert cv.value_numpy(pos, cell=np.array([10.0, 10.0, 10.0])) == pytest.approx(1.0)
