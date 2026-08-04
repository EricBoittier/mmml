"""CHARMM-compatible MM kernels: switched nonbonded, Ewald, and the ASE face.

``mm_nonbonded`` is the term the hybrid MD driver evaluates every step, so its
jitted energy and its gradient are the two numbers that set the MM half of an
MD step's cost. The Ewald classes measure the reciprocal-space sum separately
because it scales with the k-grid rather than with the pair count, and it is the
usual reason a "cheap" MM box stops being cheap.

All classes are parameterised by water count. The box is sized to liquid density
in :func:`~benchmarks._common.water_box`, so ``n_waters`` maps to a real system
size: 512 waters ≈ 1500 atoms in a 24.8 Å box, 2744 ≈ 8200 atoms in a 43.4 Å box.

Sizes start at 512 because the 12 Å production cutoff needs ``L >= 2 * cutoff``
to stay in the unique-minimum-image regime. Below that, ``_build_pair_indices``
falls back off Vesin onto chunked NumPy and the pair list degenerates to nearly
all pairs — a regime no real run is in, which would make the small end of the
scaling curve meaningless.
"""

from __future__ import annotations

import numpy as np

from ._common import block, padded_pair_list, require_jax, skip, synthetic_system

# CHARMM defaults used across the repo's condensed-phase runs.
CUTNB, CTONNB, CTOFNB = 12.0, 10.0, 12.0


def _settings(**overrides):
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings

    kwargs = dict(cutnb=CUTNB, ctonnb=CTONNB, ctofnb=CTOFNB)
    kwargs.update(overrides)
    return CharmmNbondSettings(**kwargs)


class _MMNonbondedBase:
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def _build(self, n_waters: int, *, lr_solver: str = "mic"):
        jax = require_jax()
        try:
            from mmml.md.energy import EnergyContext
            from mmml.md.energy.terms import MMNonbondedTerm
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mmml.md.energy unavailable: {exc}") from exc

        import jax.numpy as jnp

        system, box = synthetic_system(int(n_waters))
        term = MMNonbondedTerm(_settings(), lr_solver=lr_solver)
        energy_fn = term.make(system, EnergyContext()).jax_energy_fn

        pairs = padded_pair_list(system, CUTNB)
        self.n_pairs = pairs["n_pairs"]
        self.n_atoms = system.n_atoms
        kw = dict(
            pair_i=jnp.asarray(pairs["pair_i"]),
            pair_j=jnp.asarray(pairs["pair_j"]),
            pair_mask=jnp.asarray(pairs["pair_mask"]),
        )
        self.R = jnp.asarray(system.R)

        self.energy = jax.jit(lambda R: energy_fn(R, **kw))
        self.forces = jax.jit(jax.grad(lambda R: energy_fn(R, **kw)))
        block(self.energy(self.R))
        block(self.forces(self.R))


class MMNonbonded(_MMNonbondedBase):
    """Switched VDW + Coulomb (``lr_solver='mic'``) over a padded pair list."""

    params = [512, 1000, 1728, 2744]
    param_names = ["n_waters"]

    def setup(self, n_waters):
        self._build(n_waters)

    def time_energy(self, n_waters):
        block(self.energy(self.R))

    def time_forces(self, n_waters):
        block(self.forces(self.R))

    def track_pairs_in_cutoff(self, n_waters):
        """Pair count the timings above are actually over (sanity for the scaling)."""
        return self.n_pairs

    track_pairs_in_cutoff.unit = "pairs"


class MMNonbondedLongRange(_MMNonbondedBase):
    """MIC cutoff electrostatics vs. the native Ewald real+reciprocal split.

    ``ewald`` adds ``ewald_reciprocal_energy`` over a k-grid sized from the box
    and the requested accuracy, plus the self-energy correction; the pair loop is
    otherwise the same. The ratio between the two is the price of getting
    long-range electrostatics right.
    """

    params = (["mic", "ewald"], [512, 1000])
    param_names = ["lr_solver", "n_waters"]

    def setup(self, lr_solver, n_waters):
        self._build(n_waters, lr_solver=str(lr_solver))

    def time_energy(self, lr_solver, n_waters):
        block(self.energy(self.R))

    def time_forces(self, lr_solver, n_waters):
        block(self.forces(self.R))


class MMNonbondedHostReference:
    """``nonbonded_energy_and_forces`` — the host-orchestrated reference path.

    This is what the ASE calculator and the validation scripts call: it builds
    the pair list itself on every invocation, so the timing includes neighbour
    construction. That makes it the honest number for "one energy+force
    evaluation from scratch", and the gap against :class:`MMNonbonded` is the
    payoff from hoisting the pair list out of the step loop.
    """

    params = [512, 1000, 1728]
    param_names = ["n_waters"]
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self, n_waters):
        require_jax()
        try:
            from mmml.interfaces.pycharmmInterface.mm_system_energy import (
                NonbondedSystemData,
                nonbonded_energy_and_forces,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mm_system_energy unavailable: {exc}") from exc

        system, _ = synthetic_system(int(n_waters))
        ff = system.ff_params
        self.nbdata = NonbondedSystemData(
            charges=ff.charges,
            at_codes=ff.at_codes,
            epsilon=ff.epsilon,
            rmin=ff.rmin_half,
            excluded_pairs=frozenset(),
            e14_pairs=frozenset(),
        )
        self.fn = nonbonded_energy_and_forces
        self.R = np.asarray(system.R)
        self.box = np.asarray(system.box)
        self.mol_id = np.asarray(system.mol_id)
        self.settings = _settings()
        self.fn(self.R, self.nbdata, self.box, self.settings, molecule_id=self.mol_id)

    def time_energy_and_forces_from_scratch(self, n_waters):
        terms, forces = self.fn(
            self.R, self.nbdata, self.box, self.settings, molecule_id=self.mol_id
        )
        block(forces)


class EwaldReciprocal:
    """``hybrid_ewald_coulomb_energy_with_cell`` at NpT-traced cell.

    Sizes stay modest on purpose: this kernel materialises an ``(N, N, 3)`` MIC
    displacement tensor, so memory — not FLOPs — is the wall it hits first.
    """

    params = [32, 64, 128]
    param_names = ["n_waters"]
    timeout = 900.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def setup(self, n_waters):
        jax = require_jax()
        try:
            from mmml.models.ewald_hybrid_coulomb import (
                ewald_static_params_from_box_length,
                hybrid_ewald_coulomb_energy_with_cell,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"ewald_hybrid_coulomb unavailable: {exc}") from exc

        import jax.numpy as jnp

        system, box = synthetic_system(int(n_waters))
        alpha, n_int = ewald_static_params_from_box_length(box["box_L"])
        self.n_kvectors = int(np.asarray(n_int).shape[0])

        R = jnp.asarray(system.R)
        mol_id = jnp.asarray(system.mol_id)
        charges = jnp.asarray(system.ff_params.charges)
        cell = jnp.asarray(system.box)

        def energy(positions, cell_):
            return hybrid_ewald_coulomb_energy_with_cell(
                positions,
                mol_id,
                charges,
                cell_,
                alpha=alpha,
                n_int=n_int,
                n_monomers=int(n_waters),
            )

        self.R, self.cell = R, cell
        self.energy = jax.jit(energy)
        self.forces = jax.jit(jax.grad(energy, argnums=0))
        block(self.energy(self.R, self.cell))
        block(self.forces(self.R, self.cell))

    def time_energy(self, n_waters):
        block(self.energy(self.R, self.cell))

    def time_forces(self, n_waters):
        block(self.forces(self.R, self.cell))

    def track_kvectors(self, n_waters):
        return self.n_kvectors

    track_kvectors.unit = "k-vectors"
