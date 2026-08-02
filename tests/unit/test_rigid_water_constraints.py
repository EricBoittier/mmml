"""SHAKE/RATTLE constraints for rigid water on the jax-md path.

The point of these constraints is to make MD sample the geometry the ML model
was actually trained on (DES dimers are rigid: O-H = 0.9840 A, std exactly 0.0).
So the tests check the properties that matter for that: bonds really are held,
velocities carry no component along them, and energy is conserved under a
potential that would otherwise tear the molecule apart.
"""

from __future__ import annotations

import numpy as np
import pytest

# Shared by the constrained run and its unconstrained control, so the two differ
# only by the constraints. _K_BOND is the caricature of the measured ML failure:
# purely attractive along O-H, no restoring force at any separation.
_K_BOND = 20.0
_K_TRAP = 5.0
_DT = 0.25e-3
_N_STEPS = 1000


@pytest.fixture(scope="module")
def x64():
    """Enable float64, then restore whatever it was.

    Forcing it back to False leaks into later modules that need float64 -- it
    silently broke test_bonded_intra's finite-difference check when these ran in
    the same session.
    """
    import jax

    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _water(n_molecules: int, rng, r_oh=0.9572, theta_deg=104.52, jitter=0.0):
    """Ideal TIP3 geometries at random orientations/positions, optionally jittered."""
    th = np.deg2rad(theta_deg)
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh, 0.0, 0.0],
            [r_oh * np.cos(th), r_oh * np.sin(th), 0.0],
        ]
    )
    out = []
    for _ in range(n_molecules):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        out.append(base @ rot.T + rng.uniform(-8, 8, size=3))
    r = np.concatenate(out, axis=0)
    if jitter:
        r = r + rng.normal(scale=jitter, size=r.shape)
    return r


def test_tip3_geometry_is_internally_consistent() -> None:
    """r_HH is derived from the angle, so the three constraints cannot disagree."""
    from mmml.md.constraints import tip3_rigid_constraints

    spec = tip3_rigid_constraints(1)
    r_oh, theta = 0.9572, np.deg2rad(104.52)
    expected_hh = np.sqrt(2 * r_oh**2 * (1 - np.cos(theta)))
    assert spec.targets[0] == pytest.approx(r_oh)
    assert spec.targets[1] == pytest.approx(r_oh)
    assert spec.targets[2] == pytest.approx(expected_hh)
    # Mass ratio is what enters the projection; check O is heavier than H.
    assert spec.inv_mass[0] < spec.inv_mass[1]


@pytest.mark.parametrize("jitter", [0.005, 0.02, 0.05])
def test_shake_projects_step_sized_displacements(x64, jitter) -> None:
    """SHAKE is a per-step projection, so the input is one step off the manifold.

    It is not a geometry-repair tool: its correction direction comes from the
    reference bond, which only spans the constraint gradient while the
    displacement is small. 0.05 A is already generous for a 0.25 fs step.
    """
    from mmml.md.constraints import (
        constraint_residuals,
        shake_positions,
        tip3_rigid_constraints,
    )

    rng = np.random.default_rng(0)
    n = 40
    ref = _water(n, rng)
    spec = tip3_rigid_constraints(n)
    distorted = ref + rng.normal(scale=jitter, size=ref.shape)

    before = np.abs(np.asarray(constraint_residuals(distorted, spec))).max()
    fixed = shake_positions(distorted, ref, spec, iterations=200)
    after = np.abs(np.asarray(constraint_residuals(fixed, spec))).max()

    assert before > 1e-4, "test would be vacuous if the input were already rigid"
    assert after < 1e-10, f"SHAKE left residual {after:.3e} for jitter {jitter}"

    d = np.asarray(fixed).reshape(n, 3, 3)
    for a, b, target in ((1, 0, 0.9572), (2, 0, 0.9572)):
        np.testing.assert_allclose(
            np.linalg.norm(d[:, a] - d[:, b], axis=-1), target, atol=1e-6
        )


def test_shake_is_a_noop_on_an_already_rigid_configuration(x64) -> None:
    """An ideal geometry must come back bit-for-bit, or SHAKE injects drift."""
    from mmml.md.constraints import shake_positions, tip3_rigid_constraints

    rng = np.random.default_rng(1)
    r = _water(25, rng)
    spec = tip3_rigid_constraints(25)
    out = np.asarray(shake_positions(r, r, spec, iterations=50))
    np.testing.assert_allclose(out, r, atol=1e-12)


def test_rattle_removes_velocity_along_every_bond(x64) -> None:
    from mmml.md.constraints import (
        rattle_velocities,
        tip3_rigid_constraints,
    )

    rng = np.random.default_rng(2)
    n = 30
    r = _water(n, rng)
    spec = tip3_rigid_constraints(n)
    v = rng.normal(scale=0.01, size=r.shape)

    out = np.asarray(rattle_velocities(v, r, spec, iterations=200)).reshape(n, 3, 3)
    rr = r.reshape(n, 3, 3)
    for i, j in spec.pairs:
        d = rr[:, i] - rr[:, j]
        dv = out[:, i] - out[:, j]
        assert np.abs(np.sum(d * dv, axis=-1)).max() < 1e-10


def test_rattle_conserves_linear_momentum(x64) -> None:
    """Constraint impulses are internal, so total momentum must not move."""
    from mmml.md.constraints import rattle_velocities, tip3_rigid_constraints

    rng = np.random.default_rng(3)
    n = 20
    r = _water(n, rng)
    spec = tip3_rigid_constraints(n)
    v = rng.normal(scale=0.01, size=r.shape)
    m = np.tile(1.0 / spec.inv_mass, n)[:, None]

    out = np.asarray(rattle_velocities(v, r, spec, iterations=200))
    np.testing.assert_allclose((m * out).sum(axis=0), (m * v).sum(axis=0), atol=1e-10)


def test_constrained_nve_conserves_energy_under_a_bond_breaking_potential(x64) -> None:
    """The whole point, end to end.

    Integrates velocity Verlet with SHAKE on positions and RATTLE on velocities
    under a potential that is *purely attractive along O-H* -- a caricature of
    the measured ML failure, which has no restoring force and pulls bonds apart.
    Unconstrained, this dissociates the molecule and E_pot falls without bound.
    Constrained, the bonds must hold and total energy must be conserved.
    """
    import jax
    import jax.numpy as jnp

    from mmml.md.constraints import (
        constraint_residuals,
        rattle_velocities,
        shake_positions,
        tip3_rigid_constraints,
    )

    rng = np.random.default_rng(4)
    n = 8
    r0 = _water(n, rng)
    spec = tip3_rigid_constraints(n)
    inv_m = jnp.asarray(np.tile(spec.inv_mass, n)[:, None])
    mass = 1.0 / inv_m

    # Bond-breaking term plus a harmonic trap on each oxygen. Without the trap a
    # rigid molecule feels only internal forces, E_pot never changes and energy
    # conservation would hold trivially. The trap makes the molecules actually
    # move, so conservation is a real assertion about the constrained dynamics.
    def energy(pos):
        p = jnp.reshape(pos, (n, 3, 3))
        d1 = jnp.linalg.norm(p[:, 1] - p[:, 0], axis=-1)
        d2 = jnp.linalg.norm(p[:, 2] - p[:, 0], axis=-1)
        bond = -_K_BOND * jnp.sum(d1 + d2)
        trap = 0.5 * _K_TRAP * jnp.sum(p[:, 0] ** 2)
        return bond + trap

    force = jax.jit(jax.grad(lambda p: -energy(p)))
    kinetic = lambda v: 0.5 * jnp.sum(mass * v**2)

    pos = jnp.asarray(r0)
    vel = jnp.asarray(rng.normal(scale=0.002, size=r0.shape))
    vel = rattle_velocities(vel, pos, spec, iterations=100)
    dt = _DT

    @jax.jit
    def step(pos, vel):
        f = force(pos)
        vel_half = vel + 0.5 * dt * inv_m * f
        pos_new = pos + dt * vel_half
        pos_new = shake_positions(pos_new, pos, spec, iterations=30)
        # Velocity implied by the constrained move, then the second half-kick.
        vel_half = (pos_new - pos) / dt
        f_new = force(pos_new)
        vel_new = vel_half + 0.5 * dt * inv_m * f_new
        vel_new = rattle_velocities(vel_new, pos_new, spec, iterations=30)
        return pos_new, vel_new

    e0 = float(energy(pos) + kinetic(vel))
    for _ in range(_N_STEPS):
        pos, vel = step(pos, vel)
    e1 = float(energy(pos) + kinetic(vel))

    resid = float(np.abs(np.asarray(constraint_residuals(pos, spec))).max())
    assert resid < 1e-9, f"constraints drifted: {resid:.3e}"

    d = np.asarray(pos).reshape(n, 3, 3)
    oh = np.linalg.norm(d[:, 1] - d[:, 0], axis=-1)
    np.testing.assert_allclose(oh, 0.9572, atol=1e-6)

    drift = abs(e1 - e0) / max(abs(e0), 1e-12)
    assert drift < 1e-6, f"energy drift {drift:.3e} (E0={e0:.6f}, E1={e1:.6f})"


def test_unconstrained_control_actually_dissociates(x64) -> None:
    """Proves the previous test is not passing trivially.

    Identical potential, integrator, dt and step count, minus the constraints.
    The bonds must blow apart. If this ever stops dissociating, the constrained
    test is asserting nothing.
    """
    import jax
    import jax.numpy as jnp

    from mmml.md.constraints import tip3_rigid_constraints

    rng = np.random.default_rng(4)
    n = 8
    r0 = _water(n, rng)
    spec = tip3_rigid_constraints(n)
    inv_m = jnp.asarray(np.tile(spec.inv_mass, n)[:, None])

    def energy(pos):
        p = jnp.reshape(pos, (n, 3, 3))
        d1 = jnp.linalg.norm(p[:, 1] - p[:, 0], axis=-1)
        d2 = jnp.linalg.norm(p[:, 2] - p[:, 0], axis=-1)
        return -_K_BOND * jnp.sum(d1 + d2) + 0.5 * _K_TRAP * jnp.sum(p[:, 0] ** 2)

    force = jax.jit(jax.grad(lambda p: -energy(p)))
    pos = jnp.asarray(r0)
    vel = jnp.asarray(rng.normal(scale=0.002, size=r0.shape))
    for _ in range(_N_STEPS):
        f = force(pos)
        vel = vel + 0.5 * _DT * inv_m * f
        pos = pos + _DT * vel
        vel = vel + 0.5 * _DT * inv_m * force(pos)

    d = np.asarray(pos).reshape(n, 3, 3)
    oh = np.linalg.norm(d[:, 1] - d[:, 0], axis=-1)
    assert oh.max() > 1.2, f"control did not dissociate (max O-H {oh.max():.3f} A)"
