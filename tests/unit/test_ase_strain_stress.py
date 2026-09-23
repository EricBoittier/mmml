"""strain_stress_voigt (AseDimerCalculator stress) = (1/V) dE/deps, with the minimum-image lattice term and shear."""

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.mmml_calculator import strain_stress_voigt
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import cart_coords, frac_coords

L = np.array([6.0, 6.5, 7.0])
R = np.random.default_rng(0).uniform(0.0, 1.0, size=(16, 3)) * L


def _pair_energy(positions, cell):
    """Periodic soft pair energy over minimum-image pairs (range comparable to the cell)."""
    d = positions[:, None, :] - positions[None, :, :]
    s = frac_coords(d.reshape(-1, 3), cell)
    d = cart_coords(s - jax.lax.stop_gradient(jnp.round(s)), cell).reshape(d.shape)
    r2 = jnp.sum(d * d, axis=-1) + jnp.eye(positions.shape[0]) * 1e6
    return 0.5 * jnp.sum(jnp.exp(-r2 / 4.0))


def test_strain_stress_matches_finite_difference_with_shear() -> None:
    h0 = np.diag(L)
    F = -np.asarray(jax.grad(_pair_energy)(jnp.asarray(R), jnp.asarray(h0)))
    sigma = strain_stress_voigt(R, F, L, lambda cell: _pair_energy(jnp.asarray(R), cell))
    V = float(np.prod(L))
    for (a, b), k in {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (0, 2): 4, (0, 1): 5}.items():
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = 1e-6

        def e_of(sign):
            m = np.eye(3) + sign * eps
            return float(_pair_energy(jnp.asarray(R @ m.T), jnp.asarray(h0 @ m.T)))

        fd = (e_of(1) - e_of(-1)) / 2e-6 / (2.0 if a != b else 1.0)
        np.testing.assert_allclose(sigma[k] * V, fd, rtol=1e-6, atol=1e-10)


def test_central_atom_sum_alone_is_wrong_under_pbc() -> None:
    """Guard the test's sensitivity: dropping the lattice term changes the answer."""
    h0 = np.diag(L)
    F = -np.asarray(jax.grad(_pair_energy)(jnp.asarray(R), jnp.asarray(h0)))
    full = strain_stress_voigt(R, F, L, lambda cell: _pair_energy(jnp.asarray(R), cell))
    atomic_only = strain_stress_voigt(R, F, L, lambda cell: 0.0 * jnp.sum(cell))
    assert np.max(np.abs(full - atomic_only)) > 1e-3 * np.max(np.abs(full))
