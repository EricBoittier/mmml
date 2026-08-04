"""SHAKE / RATTLE holonomic constraints (``mmml.md.constraints``).

Rigid water is not optional on the ML path — the DES dimers the models are
trained on have a fixed O-H — so the constraint projection runs on every step of
every condensed-phase run. It is also a fixed-trip-count ``lax.scan``, which
means its cost does not fall when the geometry is already converged: the
``iterations`` parameter below is a straight multiplier on the per-step price,
and picking it is a real accuracy/throughput decision.

:class:`ConstrainedNVEStep` is the one to watch — it times the shipped
``apply_fn``, interleaved projections and all, against the unconstrained
``jax_md.simulate.nve`` step on the identical system, so the overhead is a ratio
rather than an absolute.
"""

from __future__ import annotations

import numpy as np

from ._common import block, require_jax, require_jax_md, skip, water_box

DT_PS = 0.5e-3


def _constraint_setup(n_molecules: int):
    try:
        from mmml.md.constraints import tip3_rigid_constraints
    except Exception as exc:  # pragma: no cover - branch-dependent
        raise skip(f"mmml.md.constraints unavailable: {exc}") from exc

    box = water_box(int(n_molecules))
    return tip3_rigid_constraints(int(n_molecules)), box


class ShakeProjection:
    """``shake_positions`` — project positions back onto the constraint manifold."""

    params = ([216, 1000], [10, 50, 100])
    param_names = ["n_waters", "iterations"]
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def setup(self, n_waters, iterations):
        jax = require_jax()
        try:
            from mmml.md.constraints import shake_positions
        except Exception as exc:  # pragma: no cover - branch-dependent
            raise skip(f"mmml.md.constraints unavailable: {exc}") from exc

        import jax.numpy as jnp

        spec, box = _constraint_setup(n_waters)
        rng = np.random.default_rng(5)
        reference = jnp.asarray(box["R"])
        # Displace by roughly one unconstrained step's worth, so the projection
        # has real work rather than converging on the first sweep.
        perturbed = jnp.asarray(box["R"] + rng.normal(scale=0.02, size=box["R"].shape))

        self.fn = jax.jit(
            lambda r: shake_positions(
                r, reference, spec, iterations=int(iterations), box=None
            )
        )
        self.perturbed = perturbed
        block(self.fn(self.perturbed))

    def time_shake(self, n_waters, iterations):
        block(self.fn(self.perturbed))


class RattleProjection:
    """``rattle_velocities`` — remove the velocity component along each bond."""

    params = ([216, 1000], [10, 50, 100])
    param_names = ["n_waters", "iterations"]
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def setup(self, n_waters, iterations):
        jax = require_jax()
        try:
            from mmml.md.constraints import rattle_velocities
        except Exception as exc:  # pragma: no cover - branch-dependent
            raise skip(f"mmml.md.constraints unavailable: {exc}") from exc

        import jax.numpy as jnp

        spec, box = _constraint_setup(n_waters)
        rng = np.random.default_rng(6)
        positions = jnp.asarray(box["R"])
        velocities = jnp.asarray(rng.normal(scale=0.01, size=box["R"].shape))

        self.fn = jax.jit(
            lambda v: rattle_velocities(
                v, positions, spec, iterations=int(iterations), box=None
            )
        )
        self.velocities = velocities
        block(self.fn(self.velocities))

    def time_rattle(self, n_waters, iterations):
        block(self.fn(self.velocities))


class ConstrainedNVEStep:
    """The shipped constrained integrator vs. plain NVE, same system, same energy.

    The potential is a cheap harmonic trap on purpose: a expensive energy would
    swamp the constraint cost and the ratio would say nothing.
    """

    params = ["unconstrained", "constrained"]
    param_names = ["integrator"]
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 30.0)

    #: steps per timed call
    n_steps = 100
    n_waters = 216

    def setup(self, integrator):
        jax = require_jax()
        require_jax_md()
        try:
            from jax_md import simulate, space

            from mmml.md.constraints import constrained_nve
        except Exception as exc:  # pragma: no cover - branch-dependent
            raise skip(f"constrained_nve unavailable: {exc}") from exc

        import jax.numpy as jnp

        spec, box = _constraint_setup(self.n_waters)
        n = self.n_waters
        mass = jnp.asarray(np.tile(1.0 / spec.inv_mass, n)[:, None])
        _displacement, shift = space.free()

        def energy(pos, **kwargs):
            p = jnp.reshape(pos, (n, 3, 3))
            return 0.5 * jnp.sum(p[:, 0] ** 2)

        if str(integrator) == "constrained":
            init_fn, apply_fn = constrained_nve(energy, shift, DT_PS, spec)
        else:
            init_fn, apply_fn = simulate.nve(energy, shift, DT_PS)

        state = init_fn(
            jax.random.PRNGKey(0), jnp.asarray(box["R"]), kT=1e-4, mass=mass
        )
        apply_jit = jax.jit(apply_fn)

        def run(state0):
            return jax.lax.fori_loop(
                0, self.n_steps, lambda _, s: apply_jit(s), state0
            )

        self.run = jax.jit(run)
        self.state = state
        block(self.run(self.state))

    def time_nve_steps(self, integrator):
        block(self.run(self.state))


class ConstraintResiduals:
    """``constraint_residuals`` — the diagnostic every constrained run logs."""

    params = [216, 1000, 4096]
    param_names = ["n_waters"]
    timeout = 300.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 10.0)

    def setup(self, n_waters):
        jax = require_jax()
        try:
            from mmml.md.constraints import constraint_residuals
        except Exception as exc:  # pragma: no cover - branch-dependent
            raise skip(f"mmml.md.constraints unavailable: {exc}") from exc

        import jax.numpy as jnp

        spec, box = _constraint_setup(n_waters)
        self.R = jnp.asarray(box["R"])
        self.fn = jax.jit(lambda r: constraint_residuals(r, spec, box=None))
        block(self.fn(self.R))

    def time_residuals(self, n_waters):
        block(self.fn(self.R))
