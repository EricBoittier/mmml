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

# mm_energy_forces.coulomb: constant * q_i q_j / r  (kcal/mol with q in e, r in A)
COULOMB_CONSTANT: float = 3.32063711e2

__all__ = [
    "COULOMB_CONSTANT",
    "RMIN_HALF_TO_SIGMA",
    "sigma_to_rmin_half",
    "cgenff_pair_lj",
    "cgenff_pair_coulomb",
    "neutralize_per_monomer",
    "cgenff_lj_energy",
    "cgenff_mm_energy",
    "monomer_centroids",
]


def sigma_to_rmin_half(sigma: Array) -> Array:
    """Dataset sigma (Angstrom) -> CHARMM Rmin/2 (Angstrom)."""
    return sigma / RMIN_HALF_TO_SIGMA


def _masked_pair_distance(
    positions: Array, iu: Array, ju: Array, mask: Array
) -> Array:
    """Pair distances with NaN-safe gradients for excluded/coincident pairs.

    ``jnp.where`` masks the *energy* but not the *gradient*: padding atoms sit on
    top of each other (r = 0), where ``d|x|/dx`` is undefined, and the NaN then
    propagates through the masked branch into every force.  Substitute a dummy
    squared distance for excluded pairs *before* the sqrt so their gradient is
    identically zero, and floor the rest away from the singularity.
    """
    d = positions[iu] - positions[ju]
    r2 = jnp.sum(d * d, axis=-1)
    r2_safe = jnp.where(mask, jnp.maximum(r2, 1e-20), 1.0)
    return jnp.sqrt(r2_safe)


def _pair_epsilon(eps: Array, iu: Array, ju: Array) -> Array:
    """Geometric-mean well depth ``sqrt(eps_i eps_j)`` with a NaN-safe gradient.

    Same hazard as :func:`_masked_pair_distance`, one operation later:
    ``d sqrt(x)/dx`` is infinite at ``x = 0`` and the cotangent of a masked-out
    pair is zero, so a zero-epsilon type (CGenFF lone pairs carry ``eps = 0`` by
    design, and padding atoms borrow row 0) yields ``0 * inf`` and poisons every
    force.  Substituting a dummy product before the sqrt makes that gradient
    identically zero instead.

    A *negative* product cannot arise from the CGenFF tables, whose epsilons are
    non-negative, only from an epsilon scale driven below zero.  Training bounds
    the scales away from that (:data:`mmml.models.mm_lj_scales.MM_LJ_EPSILON_SCALE_BOUNDS`);
    here such a pair contributes nothing rather than NaN-ing the whole system.
    """
    prod = eps[iu] * eps[ju]
    positive = prod > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, prod, 1.0)), 0.0)


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

    pair_rmin = rmin_half[iu] + rmin_half[ju]          # arithmetic (CHARMM)
    pair_eps = _pair_epsilon(eps, iu, ju)              # geometric (CHARMM)

    mask = valid[iu] & valid[ju]
    if intermolecular_only:
        mask = mask & (mol_id[iu] != mol_id[ju])

    r = _masked_pair_distance(positions, iu, ju, mask)
    e = cgenff_pair_lj(r, pair_rmin, pair_eps)
    return jnp.sum(jnp.where(mask, e, 0.0))


def cgenff_pair_coulomb(r: Array, qq: Array) -> Array:
    """Coulomb for a pair: ``332.063711 * q_i q_j / r`` (mirrors mm_energy_forces)."""
    r_safe = jnp.maximum(r, 1e-10)
    return COULOMB_CONSTANT * qq / r_safe


def neutralize_per_monomer(dq: Array, mol_id: Array, n_monomers: int = 2) -> Array:
    """Project a per-atom charge correction to be net-zero on every monomer.

    CGenFF charges are neutral per monomer, so neutral monomers interact through
    their dipoles (~1/r^3) in the far field.  PhysNet's charge head is a bare
    ``nn.Dense`` -- neutrality is only a *soft* loss penalty -- so an unprojected
    correction can leave a monomer with net charge, turning the long-range MM
    electrostatics into monopole-monopole (~1/r).  That is not a rounding error:
    a net +-0.05 e adds ~-0.10 kcal/mol at 8 A, larger than the mean |E_MM| of
    these dimers, and it decays far more slowly.

    Subtracting each monomer's mean correction preserves CGenFF's per-monomer
    neutrality while keeping the environment-dependent *shape* of the correction.
    Padding (``mol_id < 0``) is excluded and left at zero.
    """
    valid = mol_id >= 0
    dq = jnp.where(valid, dq, 0.0)
    out = jnp.zeros_like(dq)
    for m in range(n_monomers):
        sel = valid & (mol_id == m)
        n = jnp.maximum(jnp.sum(sel), 1.0)
        mean = jnp.sum(jnp.where(sel, dq, 0.0)) / n
        out = out + jnp.where(sel, dq - mean, 0.0)
    return out


def monomer_centroids(positions: Array, mol_id: Array, n_monomers: int = 2) -> Array:
    """Unweighted monomer centroids, padding-safe.

    Matches ``calculator_utils.monomer_coms_segment`` with ``masses=None`` (the
    convention the MD calculator's switching uses).
    """
    valid = (mol_id >= 0).astype(positions.dtype)
    out = []
    for m in range(n_monomers):
        w = valid * (mol_id == m).astype(positions.dtype)
        n = jnp.maximum(jnp.sum(w), 1e-10)
        out.append(jnp.sum(positions * w[:, None], axis=0) / n)
    return jnp.stack(out)


def cgenff_mm_energy(
    positions: Array,
    type_idx: Array,
    mol_id: Array,
    charges: Array,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool = True,
) -> Array:
    """Switched CGenFF MM energy (LJ + electrostatics) for one padded dimer.

    Reproduces the MD calculator's MM term: intermolecular LJ + Coulomb summed
    over atom pairs, scaled by the **same** COM-based MM taper
    (``calculator_utils.mm_switch_scale``) that the calculator applies -- the
    switching multiplies both contributions, exactly as ``_pair_vdw_energies``
    and ``coulomb`` are scaled by ``mm_scale_expanded`` there.

    Monomers (a single ``mol_id``) have no intermolecular pairs and return 0.

    Returns kcal/mol.  Padding-safe (``type_idx``/``mol_id`` < 0) and vmap-safe.
    """
    from mmml.interfaces.pycharmmInterface.calculator_utils import mm_switch_scale

    valid = type_idx >= 0
    safe_idx = jnp.where(valid, type_idx, 0)
    sig = jnp.take(master_sigmas, safe_idx)
    eps = jnp.take(master_epsilons, safe_idx)
    rmin_half = sigma_to_rmin_half(sig)
    q = jnp.where(valid, charges, 0.0)

    n = positions.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)

    pair_rmin = rmin_half[iu] + rmin_half[ju]
    pair_eps = _pair_epsilon(eps, iu, ju)
    pair_qq = q[iu] * q[ju]

    # Intermolecular pairs only; padding excluded.
    inter = valid[iu] & valid[ju] & (mol_id[iu] != mol_id[ju])

    r = _masked_pair_distance(positions, iu, ju, inter)
    pair_e = cgenff_pair_lj(r, pair_rmin, pair_eps) + cgenff_pair_coulomb(r, pair_qq)
    e_raw = jnp.sum(jnp.where(inter, pair_e, 0.0))

    coms = monomer_centroids(positions, mol_id, n_monomers=2)
    d_com = coms[1] - coms[0]
    r_com = jnp.sqrt(jnp.maximum(jnp.sum(d_com * d_com), 1e-20))
    scale = mm_switch_scale(
        r_com,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        ml_switch_width=ml_switch_width,
        complementary_handoff=complementary_handoff,
    )
    # No inter pairs (monomer) => 0 regardless of the (meaningless) centroid sep.
    has_inter = jnp.any(inter)
    return jnp.where(has_inter, scale * e_raw, 0.0)
