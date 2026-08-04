"""SHAKE/RATTLE holonomic distance constraints for jax-md hybrids.

Why this exists
---------------
The jax-md path has no SHAKE -- the CHARMM path runs ``shake bonh para sele all
end`` but the jax-md runner has nothing equivalent. For a hybrid whose ML term
was trained on rigid monomers that is not a detail: the DES training dimers are
perfectly rigid (O-H = 0.9840 A, std exactly 0.0), so the ML monomer term carries
no restoring force, bulk water tears itself apart internally, and E_pot falls
without bound. See ``docs/hybrid-bonded-intra.md``.

Constraining the monomers makes MD sample exactly the distribution the model was
trained on, which is also how rigid TIP3P is normally run in CHARMM.

What this implements
--------------------
Holonomic distance constraints on atom pairs within each molecule:

  SHAKE   positions   |r_i - r_j| = d_ij
  RATTLE  velocities  (v_i - v_j) . (r_i - r_j) = 0

Both are mass-weighted and solved by fixed-count Gauss-Seidel sweeps, so the
whole thing stays jit-friendly with no data-dependent control flow.

Rigid TIP3 water is three constraints on three atoms -- O-H1, O-H2 and the H1-H2
pseudo-bond that CHARMM's TIP3 residue carries for exactly this purpose. Those
three fix all internal degrees of freedom, so no separate angle constraint is
needed.

Units are ASE/jax-md house units: Angstrom, and masses in whatever unit the
caller uses consistently (only mass ratios enter).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "MolecularConstraints",
    "tip3_rigid_constraints",
    "shake_positions",
    "rattle_velocities",
    "constraint_residuals",
    "wrap_apply_fn_with_constraints",
    "molecular_virial_decomposition",
]


@dataclass(frozen=True)
class MolecularConstraints:
    """Distance constraints repeated identically across ``n_molecules``.

    ``pairs`` indexes atoms *within* a molecule, so one spec covers every
    molecule of that species. ``targets`` are the constrained distances in A.
    ``inv_mass`` is 1/m per atom within the molecule.
    """

    pairs: np.ndarray  # (n_constraints, 2) int
    targets: np.ndarray  # (n_constraints,) float, Angstrom
    inv_mass: np.ndarray  # (atoms_per_molecule,) float
    atoms_per_molecule: int
    n_molecules: int

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must be (n_constraints, 2), got {pairs.shape}")
        if pairs.max(initial=-1) >= self.atoms_per_molecule:
            raise ValueError(
                f"pair index {pairs.max()} outside a molecule of "
                f"{self.atoms_per_molecule} atoms"
            )
        if len(np.asarray(self.targets)) != len(pairs):
            raise ValueError("targets and pairs must have the same length")
        if len(np.asarray(self.inv_mass)) != self.atoms_per_molecule:
            raise ValueError("inv_mass must have one entry per atom in a molecule")
        if np.any(np.asarray(self.targets) <= 0.0):
            raise ValueError("constraint distances must be positive")


def tip3_rigid_constraints(
    n_molecules: int,
    *,
    r_oh: float = 0.9572,
    theta_hoh_deg: float = 104.52,
    mass_o: float = 15.9994,
    mass_h: float = 1.008,
) -> MolecularConstraints:
    """Rigid TIP3 water: O-H1, O-H2, and the H1-H2 pseudo-bond.

    Defaults are the CHARMM TIP3 geometry. ``r_hh`` is derived from the angle by
    the law of cosines rather than taken as an independent parameter, so the
    three constraints cannot be mutually inconsistent.

    Atom order is assumed O, H, H within each molecule, which is what the
    hybrid's monomer slicing already assumes elsewhere.
    """
    theta = np.deg2rad(float(theta_hoh_deg))
    r_hh = float(np.sqrt(2.0 * r_oh**2 * (1.0 - np.cos(theta))))
    return MolecularConstraints(
        pairs=np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        targets=np.array([r_oh, r_oh, r_hh], dtype=np.float64),
        inv_mass=np.array(
            [1.0 / mass_o, 1.0 / mass_h, 1.0 / mass_h], dtype=np.float64
        ),
        atoms_per_molecule=3,
        n_molecules=int(n_molecules),
    )


def _as_molecules(x, spec: MolecularConstraints):
    return jnp.reshape(x, (spec.n_molecules, spec.atoms_per_molecule, 3))


def constraint_residuals(positions, spec: MolecularConstraints):
    """``|r_i - r_j|^2 - d_ij^2`` per molecule per constraint.

    Squared form, matching what SHAKE actually drives to zero.
    """
    r = _as_molecules(positions, spec)
    i = jnp.asarray(spec.pairs[:, 0])
    j = jnp.asarray(spec.pairs[:, 1])
    d = r[:, i, :] - r[:, j, :]
    return jnp.sum(d * d, axis=-1) - jnp.asarray(spec.targets) ** 2


def shake_positions(
    positions,
    reference,
    spec: MolecularConstraints,
    *,
    iterations: int = 100,
    tolerance: float = 1e-10,
):
    """Project ``positions`` onto the constraint manifold (SHAKE).

    ``reference`` is the pre-step, already-satisfying configuration whose bond
    vectors define the correction directions -- this is what makes SHAKE a
    projection along the constraint gradients rather than an arbitrary
    rearrangement. Passing ``positions`` as its own reference degrades accuracy
    but still converges for small displacements.

    ``iterations`` is a fixed trip count so the sweep stays jittable;
    ``tolerance`` only gates whether a sweep still applies a correction, it does
    not terminate the loop early.
    """
    # Constraint indices are static (three for rigid water), so the inner loop is
    # a Python loop. Tracing it with scan would make i/j tracers, which cannot
    # index .at[:, i, :].
    pairs = [(int(a), int(b)) for a, b in np.asarray(spec.pairs)]
    targets_sq = np.asarray(spec.targets, dtype=np.float64) ** 2
    inv_mass = np.asarray(spec.inv_mass, dtype=np.float64)

    r_ref = _as_molecules(reference, spec)

    def sweep(r_cur, _):
        for c, (i, j) in enumerate(pairs):
            d_ref = r_ref[:, i, :] - r_ref[:, j, :]
            d_cur = r_cur[:, i, :] - r_cur[:, j, :]
            resid = jnp.sum(d_cur * d_cur, axis=-1) - targets_sq[c]
            denom = 2.0 * (inv_mass[i] + inv_mass[j]) * jnp.sum(d_cur * d_ref, axis=-1)
            # A vanishing denominator means the current and reference bonds are
            # orthogonal; no correction is representable along d_ref, so skip
            # rather than divide.
            safe = jnp.abs(denom) > 1e-12
            lam = jnp.where(safe, resid / jnp.where(safe, denom, 1.0), 0.0)
            lam = jnp.where(jnp.abs(resid) > tolerance, lam, 0.0)
            corr = lam[:, None] * d_ref
            r_cur = r_cur.at[:, i, :].add(-inv_mass[i] * corr)
            r_cur = r_cur.at[:, j, :].add(+inv_mass[j] * corr)
        return r_cur, None

    r, _ = jax.lax.scan(sweep, _as_molecules(positions, spec), None, length=int(iterations))
    return jnp.reshape(r, (-1, 3))


def rattle_velocities(
    velocities,
    positions,
    spec: MolecularConstraints,
    *,
    iterations: int = 100,
    tolerance: float = 1e-12,
):
    """Project ``velocities`` so each constrained bond has no rate of change.

    Enforces ``(v_i - v_j) . (r_i - r_j) = 0``. Without this, positions stay
    rigid but the velocities retain components along the bonds, so the reported
    kinetic energy and temperature are wrong and the constraint forces do work.
    """
    pairs = [(int(a), int(b)) for a, b in np.asarray(spec.pairs)]
    inv_mass = np.asarray(spec.inv_mass, dtype=np.float64)

    r = _as_molecules(positions, spec)

    def sweep(v_cur, _):
        for i, j in pairs:
            d = r[:, i, :] - r[:, j, :]
            dv = v_cur[:, i, :] - v_cur[:, j, :]
            rv = jnp.sum(d * dv, axis=-1)
            denom = (inv_mass[i] + inv_mass[j]) * jnp.sum(d * d, axis=-1)
            safe = denom > 1e-12
            lam = jnp.where(safe, rv / jnp.where(safe, denom, 1.0), 0.0)
            lam = jnp.where(jnp.abs(rv) > tolerance, lam, 0.0)
            corr = lam[:, None] * d
            v_cur = v_cur.at[:, i, :].add(-inv_mass[i] * corr)
            v_cur = v_cur.at[:, j, :].add(+inv_mass[j] * corr)
        return v_cur, None

    v, _ = jax.lax.scan(sweep, _as_molecules(velocities, spec), None, length=int(iterations))
    return jnp.reshape(v, (-1, 3))


def wrap_apply_fn_with_constraints(
    apply_fn,
    spec: MolecularConstraints,
    *,
    shake_iterations: int = 100,
    rattle_iterations: int = 100,
):
    """Compose SHAKE/RATTLE onto a jax-md ``apply_fn``.

    Wrapping rather than editing the integrator keeps this out of
    ``set_up_nhc_sim_routine``, which is already over the oversized-function
    ratchet.

    The projection is applied at step boundaries: positions first, using the
    pre-step configuration as the SHAKE reference, then velocities against the
    constrained positions. jax-md states carry ``momentum``, so the velocity
    projection converts both ways around ``mass``.

    Note this is step-boundary projection, not RATTLE interleaved inside the
    velocity-Verlet half-kicks. For small ``dt`` the two agree; the NVE
    conservation gate is what decides whether ``dt`` is small enough, so run it
    rather than assuming.
    """
    from jax_md import dataclasses as jax_md_dataclasses

    def constrained_apply(state, **kwargs):
        previous = state.position
        state = apply_fn(state, **kwargs)

        position = shake_positions(
            state.position, previous, spec, iterations=shake_iterations
        )
        mass = state.mass
        velocity = rattle_velocities(
            state.momentum / mass, position, spec, iterations=rattle_iterations
        )
        return jax_md_dataclasses.replace(
            state, position=position, momentum=velocity * mass
        )

    return constrained_apply


def molecular_virial_decomposition(positions, forces, spec: MolecularConstraints):
    """Split the virial into intermolecular (molecular) and internal parts.

    For rigid molecules the atomic virial ``sum_i r_i . f_i`` and the molecular
    virial ``sum_mol R_com . F_mol`` differ by the work of the internal forces,
    including the constraint forces. Only the molecular form is meaningful for
    the pressure of a constrained system -- the atomic form picks up constraint
    contributions that do no work and should not appear in P.

    This matters here: the NpT virial was previously measured as identically
    zero (pressure matched 2KE/3V to 0.001%), so the decomposition is worth
    looking at directly rather than trusting a single scalar.

    Returns ``(w_atomic, w_molecular, w_internal)`` with
    ``w_internal = w_atomic - w_molecular``, each a scalar in energy units.
    """
    r = _as_molecules(positions, spec)
    f = _as_molecules(forces, spec)
    mass = jnp.asarray(1.0 / spec.inv_mass)[None, :, None]

    w_atomic = jnp.sum(r * f)
    com = jnp.sum(r * mass, axis=1) / jnp.sum(mass, axis=1)
    f_mol = jnp.sum(f, axis=1)
    w_molecular = jnp.sum(com * f_mol)
    return w_atomic, w_molecular, w_atomic - w_molecular


def rigid_water_spec_from_args(args, n_monomers: int, monomer_offsets) -> "MolecularConstraints | None":
    """Build the TIP3 constraint spec from CLI args, or ``None`` when disabled.

    Lives here rather than in the runner so ``set_up_nhc_sim_routine`` does not
    grow -- it is already over the oversized-function ratchet.

    Refuses heterogeneous monomers rather than silently constraining the wrong
    atoms: the spec is one pattern repeated across molecules, so a mixed system
    would apply water constraints to whatever happens to sit at those offsets.
    """
    if not bool(getattr(args, "rigid_water", False)):
        return None

    offsets = np.asarray(monomer_offsets, dtype=int)
    sizes = np.diff(offsets)
    unique = sorted(set(int(s) for s in sizes))
    if unique != [3]:
        raise NotImplementedError(
            f"--rigid-water expects 3-atom monomers throughout; found sizes {unique}. "
            "Constraining a mixed system with one repeated pattern would apply "
            "water constraints to whatever sits at those offsets."
        )
    return tip3_rigid_constraints(
        int(n_monomers),
        r_oh=float(getattr(args, "rigid_water_roh", 0.9572)),
        theta_hoh_deg=float(getattr(args, "rigid_water_theta", 104.52)),
    )


def maybe_wrap_rigid_water(apply_fn, args, n_monomers: int, monomer_offsets, console=None):
    """Compose rigid-water constraints onto ``apply_fn`` when ``--rigid-water`` is set.

    Returns ``apply_fn`` untouched when the flag is absent, so existing runs are
    bit-identical. Reporting lives here too: ``set_up_nhc_sim_routine`` is over
    the oversized-function ratchet, so the call site stays one line.
    """
    spec = rigid_water_spec_from_args(args, n_monomers, monomer_offsets)
    if spec is None:
        return apply_fn
    if console is not None:
        try:
            from rich.panel import Panel

            console.print(
                Panel(
                    f"SHAKE/RATTLE on {n_monomers} monomers | "
                    f"O-H={spec.targets[0]:.4f} A  H-H={spec.targets[2]:.4f} A",
                    title="[bold]Rigid water[/bold]",
                    border_style="cyan",
                )
            )
        except Exception:
            pass
    return wrap_apply_fn_with_constraints(apply_fn, spec)
