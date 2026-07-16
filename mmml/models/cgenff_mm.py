"""CGenFF MM (Lennard-Jones) for hybrid ML/MM training -- no CHARMM required.

The MD hybrid calculator evaluates MM through
``mm_energy_forces.build_mm_energy_forces_fn``, which is bound to a live
pycharmm session (its ``at_codes`` index ``param.get_atc()``) and is built for
one fixed system stepped many times.  Training needs the opposite: many small,
heterogeneous, padded structures evaluated once each, with no CHARMM in the
loop.

Rather than reuse that plumbing, this module reproduces the MD **formula**
exactly while sourcing parameters from the dataset's own CGenFF fields
(``cgenff_type_idx`` + ``cgenff_master_sigmas``/``cgenff_master_epsilons``).
That is safe because the parameters are pinned equal to CHARMM's by
``tests/unit/test_cgenff_lj_parity.py``, and the formula is pinned to the MD
math by ``tests/unit/test_cgenff_mm_energy.py``.

Conventions (see the parity test):

* dataset stores ``sigma`` (Angstrom) and ``epsilon >= 0``
* CHARMM/``mm_energy_forces`` uses ``Rmin/2`` (Angstrom) and forces
  ``epsilon <= 0`` via ``-abs(eps)``

The MD combining rules are ``Rmin_ij = Rmin_i/2 + Rmin_j/2`` (arithmetic) and
``eps_ij = sqrt(eps_i * eps_j)`` (geometric).  The per-atom epsilon *sign
cancels* in that geometric mean, so the dataset's positive epsilon yields an
identical ``eps_ij``; only the length convention needs converting.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

# sigma = 2 * (Rmin/2) / 2^(1/6)   <->   Rmin/2 = sigma / (2 / 2^(1/6))
RMIN_HALF_TO_SIGMA: float = 2.0 / (2.0 ** (1.0 / 6.0))

__all__ = [
    "RMIN_HALF_TO_SIGMA",
    "sigma_to_rmin_half",
    "cgenff_pair_lj",
    "cgenff_lj_energy",
]


def sigma_to_rmin_half(sigma: Array) -> Array:
    """Dataset sigma (Angstrom) -> CHARMM Rmin/2 (Angstrom)."""
    return sigma / RMIN_HALF_TO_SIGMA


def cgenff_pair_lj(r: Array, pair_rmin: Array, pair_eps: Array) -> Array:
    """CHARMM Lennard-Jones for a pair: ``eps * [(Rmin/r)^12 - 2 (Rmin/r)^6]``.

    Mirrors ``mm_energy_forces.lennard_jones`` exactly, including its
    ``r_safe`` guard.  ``pair_eps`` is the *positive* well depth (the geometric
    mean of the per-atom epsilons), ``pair_rmin`` is ``Rmin_i/2 + Rmin_j/2``.
    """
    r_safe = jnp.maximum(r, 1e-10)
    r6 = (pair_rmin / r_safe) ** 6
    return pair_eps * (r6**2 - 2.0 * r6)


def cgenff_lj_energy(
    positions: Array,
    type_idx: Array,
    mol_id: Array,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    intermolecular_only: bool = True,
) -> Array:
    """Total CGenFF LJ energy for one (padded) structure.

    Parameters
    ----------
    positions : (n_atoms, 3)
    type_idx : (n_atoms,) index into the master tables; ``< 0`` marks padding.
    mol_id : (n_atoms,) monomer id; ``< 0`` marks padding.
    master_sigmas, master_epsilons : (n_types,) dataset tables (sigma, eps>=0).
    intermolecular_only : exclude pairs within the same monomer (their bonded
        terms are not part of this MM residual), matching the hybrid decomposition.

    Returns
    -------
    Scalar energy in the master tables' units (kcal/mol for CGenFF).

    All-pairs: the hybrid training structures are small (<= 20 atoms), so no
    neighbour list is needed.  Padding-safe and vmap-safe (static shapes).
    """
    valid = type_idx >= 0
    safe_idx = jnp.where(valid, type_idx, 0)

    sig = jnp.take(master_sigmas, safe_idx)
    eps = jnp.take(master_epsilons, safe_idx)
    rmin_half = sigma_to_rmin_half(sig)

    n = positions.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)

    d = positions[iu] - positions[ju]
    r = jnp.linalg.norm(d, axis=-1)

    pair_rmin = rmin_half[iu] + rmin_half[ju]          # arithmetic (CHARMM)
    pair_eps = jnp.sqrt(eps[iu] * eps[ju])             # geometric (CHARMM)

    mask = valid[iu] & valid[ju]
    if intermolecular_only:
        mask = mask & (mol_id[iu] != mol_id[ju])

    e = cgenff_pair_lj(r, pair_rmin, pair_eps)
    return jnp.sum(jnp.where(mask, e, 0.0))
