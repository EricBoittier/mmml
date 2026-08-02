"""The NpT energy's custom VJP must return the virial for `perturbation`.

jax-md gets the internal pressure by differentiating the energy with respect to
an isotropic volume perturbation::

    U(eps) = energy_fn(..., perturbation=1 + eps)
    dU_dV  ~ grad(U)(0.0)

``jaxmd_runner.set_up_nhc_sim_routine`` wraps the hybrid energy in a
``jax.custom_vjp`` because plain ``jax.grad`` through the calculator gives NaN.
That backward pass used to return ``None`` for the ``perturbation`` cotangent,
which makes the derivative identically zero and leaves the barostat with only
the kinetic term::

    P_int = 2 * KE / (3 V) + 0

Measured, not inferred: a 732-TIP3 box at 297.87 K in 21955.3 A^3 reported
P_meas = 4059.58 atm against a 1 atm target; the kinetic-only value for that
state is 4059.63 atm, agreeing to 0.001%. The barostat then drove the cell on a
4000x pressure error and the run blew up to 8e7 eV.

These tests pin the analytic virial

    dE/dp = -(1 / 3p) * sum_i F_i . r_i

against finite differences of the energy, on potentials whose virial is known
independently. They deliberately avoid importing the runner (which needs CHARMM
and a model) and instead reproduce the exact VJP contract in miniature.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _make_pair_energy():
    """Sum of r^-12 over all pairs -- virial known analytically."""

    def energy_of_positions(r):
        d = r[:, None, :] - r[None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        n = r.shape[0]
        r2 = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, r2)
        return 0.5 * jnp.sum(r2 ** -6)

    return energy_of_positions


def _scaled_energy(energy_of_positions, r0, p):
    """Energy after isotropic scaling r = p^(1/3) r0 -- the perturbation path."""
    return energy_of_positions(jnp.cbrt(p) * r0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_virial_matches_finite_difference(seed):
    energy_of_positions = _make_pair_energy()
    key = jax.random.PRNGKey(seed)
    r0 = jax.random.uniform(key, (8, 3), minval=0.5, maxval=3.0, dtype=jnp.float64)

    p = 1.0
    r = jnp.cbrt(p) * r0
    # F = -dE/dr, exactly what the runner's force fn supplies.
    F = -jax.grad(energy_of_positions)(r)
    analytic = -jnp.sum(F * r) / (3.0 * p)

    fd = jax.grad(lambda pp: _scaled_energy(energy_of_positions, r0, pp))(p)

    assert float(analytic) == pytest.approx(float(fd), rel=1e-6)


def test_virial_is_nonzero_for_a_real_configuration():
    """Guards the actual regression: a zero cotangent would also 'pass' a
    sloppy comparison, so assert the virial is materially non-zero."""
    energy_of_positions = _make_pair_energy()
    key = jax.random.PRNGKey(3)
    r = jax.random.uniform(key, (10, 3), minval=0.6, maxval=2.5, dtype=jnp.float64)
    F = -jax.grad(energy_of_positions)(r)
    virial_term = -jnp.sum(F * r) / 3.0
    assert abs(float(virial_term)) > 1e-6


def test_custom_vjp_contract_propagates_the_perturbation_cotangent():
    """Reproduce the runner's custom_vjp shape and check grad reaches it."""
    energy_of_positions = _make_pair_energy()
    key = jax.random.PRNGKey(5)
    r0 = jax.random.uniform(key, (6, 3), minval=0.6, maxval=2.5, dtype=jnp.float64)

    @jax.custom_vjp
    def energy(frac, perturbation=None):
        return _scaled_energy(energy_of_positions, frac, perturbation)

    def fwd(frac, perturbation):
        return energy(frac, perturbation), (frac, perturbation)

    def bwd(res, g):
        frac, perturbation = res
        r = jnp.cbrt(perturbation) * frac
        F = -jax.grad(energy_of_positions)(r)
        grad_frac = -F * g
        grad_pert = -jnp.sum(F * r) / (3.0 * perturbation) * g
        return (grad_frac, grad_pert)

    energy.defvjp(fwd, bwd)

    got = jax.grad(lambda p: energy(r0, p))(1.0)
    want = jax.grad(lambda p: _scaled_energy(energy_of_positions, r0, p))(1.0)
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    assert abs(float(got)) > 1e-6, "cotangent collapsed to zero -- the bug"


def test_a_none_cotangent_would_have_been_caught():
    """Explicitly show the failing behaviour this test file exists to prevent."""
    energy_of_positions = _make_pair_energy()
    key = jax.random.PRNGKey(9)
    r0 = jax.random.uniform(key, (6, 3), minval=0.6, maxval=2.5, dtype=jnp.float64)

    @jax.custom_vjp
    def broken(frac, perturbation=None):
        return _scaled_energy(energy_of_positions, frac, perturbation)

    def fwd(frac, perturbation):
        return broken(frac, perturbation), (frac, perturbation)

    def bwd(res, g):
        frac, perturbation = res
        r = jnp.cbrt(perturbation) * frac
        F = -jax.grad(energy_of_positions)(r)
        return (-F * g, jnp.zeros_like(perturbation))  # the old None -> zero

    broken.defvjp(fwd, bwd)

    got = jax.grad(lambda p: broken(r0, p))(1.0)
    want = jax.grad(lambda p: _scaled_energy(energy_of_positions, r0, p))(1.0)
    assert float(got) == 0.0
    assert abs(float(want)) > 1e-6
    assert float(got) != pytest.approx(float(want), rel=1e-6)
