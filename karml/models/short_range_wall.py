"""Short-range inter-monomer repulsion: a safety net outside the training data.

The hybrid handoff switches the MM Lennard-Jones wall **off** below
``mm_switch_on - ml_switch_width`` (6.5 A by default) and hands the close range
to the ML model.  That is correct where the model has data -- but the model has
no repulsive prior *outside* it, so nothing stops atoms collapsing.  Seen in a
liquid acetone run: two monomers reached an atom-atom separation of 0.28 A,
requiring a CHARMM SD/ABNR rescue.

Why atom-pair and not COM-COM.  Measured over the 5785 real training dimers
(``out_combined_dedup``):

* smallest COM separation ever sampled : 3.47 A
* closest inter-monomer atom contact   : 1.97 A

A wall must be ~0 everywhere the data lives, so a COM wall could only act below
~3.4 A.  But DCM's chlorines sit ~1.77 A off the COM, so two monomers at a COM
separation of 4.0 A -- legal, sampled, and untouchable by any COM wall -- can
still have atoms at 4.0 - 1.77 - 1.77 = 0.46 A.  An off-axis glancing approach
produces exactly the observed 0.28 A contact while the COMs never come close.
A COM wall is therefore *geometrically incapable* of preventing this failure;
an atom-pair wall is, and it subsumes the COM case (atoms that cannot overlap
cannot let monomers merge).

Placement.  ``r_on`` must sit **below** chemically normal intermolecular
contacts (water H-bonds: H···O ~1.5–1.9 A) and **above** the ZBL prior
(cutoff 0.6 A).  An earlier 1.9 A onset was just under the DCM training
minimum (1.97 A) but actively pushed on liquid-water H-bonds (~10³ pairs / box
inside the wall).  ``r_on = 1.0 A`` is identically zero on every training
dimer and on normal H-bonded liquids; it only catches trajectories that leave
that region.  ZBL still owns the deep core.

Form::

    E(r) = k * (r_on - r)^3 / r     for r < r_on;   0 for r >= r_on

* value, first and second derivative all vanish at ``r_on`` -> C2 continuous,
  no force discontinuity where it switches on;
* diverges as 1/r -> genuinely impenetrable, unlike a bare cubic which
  saturates at a finite height an energetic atom can tunnel through.

Units are **eV / eV per Angstrom** (the canonical hybrid-inference units, see
``mmml/data/units.py``).  Callers working in kcal/mol must convert.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

__all__ = [
    "DEFAULT_WALL_K_EV_A2",
    "DEFAULT_WALL_R_ON_A",
    "pair_wall_energy",
    "inter_monomer_wall_energy",
]

#: Onset (Angstrom).  Below water H-bond contacts (~1.5 A) and the 1.971 A
#: closest inter-monomer contact in the DCM training data, and above the ZBL
#: cutoff (0.6 A).  Identically zero on every training structure and on
#: chemically normal liquid contacts.
DEFAULT_WALL_R_ON_A: float = 1.0

#: Stiffness (eV * Angstrom^2).  Scaled with the lower onset so the observed
#: 0.28 A collapse still costs >100 eV (k=80 -> ~105 eV at 0.28 A), while the
#: wall remains exactly 0 at and above ``r_on``.
DEFAULT_WALL_K_EV_A2: float = 80.0


def pair_wall_energy(
    r: Array,
    *,
    r_on: float = DEFAULT_WALL_R_ON_A,
    k: float = DEFAULT_WALL_K_EV_A2,
) -> Array:
    """``k * (r_on - r)^3 / r`` below ``r_on``, exactly 0 at and above it (eV)."""
    # Guard the 1/r divergence before it is evaluated: r=0 gives inf, and an inf
    # inside the untaken branch of a where() still poisons the gradient.
    r_safe = jnp.maximum(r, 1e-6)
    inside = r_safe < r_on
    d = jnp.where(inside, r_on - r_safe, 0.0)
    return jnp.where(inside, k * d**3 / r_safe, 0.0)


def inter_monomer_wall_energy(
    positions: Array,
    mol_id: Array,
    *,
    r_on: float = DEFAULT_WALL_R_ON_A,
    k: float = DEFAULT_WALL_K_EV_A2,
) -> Array:
    """Total wall energy (eV) for one padded structure.

    Sums :func:`pair_wall_energy` over inter-monomer atom pairs only.  Pairs
    inside a monomer are excluded -- their bonded geometry lives at 1.0-1.5 A,
    which can sit near or inside ``r_on``, and is the model's business, not the
    wall's.

    ``mol_id < 0`` marks padding and is excluded.  Monomers (a single
    ``mol_id``) have no inter-monomer pairs and return 0.  Padding-safe and
    vmap-safe (static shapes).
    """
    valid = mol_id >= 0
    n = positions.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    inter = valid[iu] & valid[ju] & (mol_id[iu] != mol_id[ju])

    d = positions[iu] - positions[ju]
    r2 = jnp.sum(d * d, axis=-1)
    # Substitute a dummy distance for excluded pairs BEFORE the sqrt: padding
    # atoms are coincident (r=0) where d|x|/dx is undefined, and masking the
    # energy afterwards leaves 0 * NaN = NaN in every force.
    r2_safe = jnp.where(inter, jnp.maximum(r2, 1e-20), 1.0)
    r = jnp.sqrt(r2_safe)

    e = pair_wall_energy(r, r_on=r_on, k=k)
    return jnp.sum(jnp.where(inter, e, 0.0))
