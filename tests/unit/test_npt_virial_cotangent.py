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


# --------------------------------------------------------------------------
# Independent references. The tests above compare the analytic virial against
# finite differences of the *same* energy function, which catches a wrong sign
# or factor but would not catch a shared misconception. These compare against
# results derived independently of the code under test.
# --------------------------------------------------------------------------


def test_virial_of_an_inverse_power_law_matches_euler_theorem():
    """For U = sum C r^-n, Euler's theorem gives the virial in closed form.

    U is homogeneous of degree -n in the coordinates, so sum_i r_i . dU/dr_i =
    -n U exactly. Hence sum_i F_i . r_i = +n U and

        dE/dp at p=1  =  -(1/3) sum F.r  =  -(n/3) U

    This is a property of the potential, not of our differentiation.
    """
    key = jax.random.PRNGKey(17)
    r = jax.random.uniform(key, (9, 3), minval=0.7, maxval=2.5, dtype=jnp.float64)

    for n in (6, 12):
        def U(x, n=n):
            d = x[:, None, :] - x[None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            m = x.shape[0]
            r2 = jnp.where(jnp.eye(m, dtype=bool), jnp.inf, r2)
            return 0.5 * jnp.sum(r2 ** (-n / 2))

        F = -jax.grad(U)(r)
        analytic = -jnp.sum(F * r) / 3.0
        euler = -(n / 3.0) * U(r)
        assert float(analytic) == pytest.approx(float(euler), rel=1e-8)


def test_virial_vanishes_at_a_potential_minimum_pair():
    """Two atoms at the LJ minimum have zero pair force, hence zero virial."""
    sigma, eps = 3.405, 0.238  # argon
    rmin = 2.0 ** (1.0 / 6.0) * sigma

    def U(x):
        d = x[1] - x[0]
        r = jnp.linalg.norm(d)
        sr6 = (sigma / r) ** 6
        return 4.0 * eps * (sr6 * sr6 - sr6)

    r = jnp.array([[0.0, 0.0, 0.0], [rmin, 0.0, 0.0]], dtype=jnp.float64)
    F = -jax.grad(U)(r)
    assert float(jnp.abs(F).max()) == pytest.approx(0.0, abs=1e-9)
    assert float(-jnp.sum(F * r) / 3.0) == pytest.approx(0.0, abs=1e-9)


def test_ideal_gas_pressure_identity_reproduces_the_observed_blowup_number():
    """With zero virial the pressure collapses to 2KE/(3V) -- the 4059.58 atm.

    This is the arithmetic that identified the bug, pinned so the reasoning
    cannot silently rot: the NpT log reported P_meas = 4059.58 atm for a state
    at T = 297.87 K, V = 21955.3 A^3, 2196 atoms, against a 1 atm target.
    """
    kB = 8.617333262e-5           # eV/K
    ev_a3_to_pa = 1.602176634e-19 / 1e-30
    atm = 101325.0

    T, V, n_atoms = 297.87, 21955.3, 2196
    ke = 0.5 * (3 * n_atoms) * kB * T
    p_kin_atm = (2.0 * ke / (3.0 * V)) * ev_a3_to_pa / atm

    assert p_kin_atm == pytest.approx(4059.58, rel=2e-5), (
        "kinetic-only pressure no longer reproduces the observed P_meas; "
        "the zero-virial diagnosis rests on this identity"
    )


def test_virial_scales_linearly_with_a_uniform_energy_scale():
    """Doubling the potential doubles the virial -- a basic linearity check."""
    key = jax.random.PRNGKey(23)
    r = jax.random.uniform(key, (7, 3), minval=0.7, maxval=2.0, dtype=jnp.float64)

    def U(x, c=1.0):
        d = x[:, None, :] - x[None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        m = x.shape[0]
        r2 = jnp.where(jnp.eye(m, dtype=bool), jnp.inf, r2)
        return c * 0.5 * jnp.sum(r2 ** -6)

    v1 = -jnp.sum(-jax.grad(lambda x: U(x, 1.0))(r) * r) / 3.0
    v2 = -jnp.sum(-jax.grad(lambda x: U(x, 2.0))(r) * r) / 3.0
    assert float(v2) == pytest.approx(2.0 * float(v1), rel=1e-9)
