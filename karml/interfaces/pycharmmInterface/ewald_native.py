"""Jit-native, fully differentiable Ewald summation for periodic Coulomb energy.

Exists because the ``jax_pme``/``nvalchemiops_pme`` backends (see
``long_range_backend.py``) are host-orchestrated (they build an ``ase.Atoms``
and call out via ``jax.pure_callback``), which cannot sit inside the
``jax.grad``-differentiated energy function that ``jax_md``'s integrators
require -- ``jax.pure_callback`` has no VJP. This module implements the
standard Ewald split directly in JAX so the reciprocal-space sum is just
another jittable, autodiff-through-able reduction, usable from
``mm_nonbonded.py``'s jax/jit face (``lr_solver="ewald"``).

Convention: energies returned here are in *raw* Coulomb units (``qq/r``, no
Coulomb constant) -- exactly like ``mm_system_energy._pair_elec_energy``'s
``raw`` branch -- so callers multiply the total by ``COULOMB_KCAL`` (kcal/mol)
themselves, same as every other electrostatic term in this codebase.

Standard formulas (Gaussian units, splitting parameter ``alpha``, box volume
``V``, reciprocal vectors ``k``, structure factor ``S(k) = sum_i q_i
exp(i k . r_i)``)::

    E_real  = sum_{i<j, r_ij < rcut} q_i q_j erfc(alpha r_ij) / r_ij
    E_recip = (2*pi/V) * sum_{k != 0} exp(-k^2/(4 alpha^2)) / k^2 * |S(k)|^2
    E_self  = -(alpha/sqrt(pi)) * sum_i q_i^2
    E_excl  = -sum_{(i,j) excluded} q_i q_j erf(alpha r_ij) / r_ij

``E_excl`` is required because the reciprocal sum implicitly includes every
pair (including bonded 1-2/1-3 exclusions); it subtracts the reciprocal-space
contribution for pairs that should not interact at all, leaving the intended
net-zero.

References: Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., ch.
12; Essmann et al., J. Chem. Phys. 103, 8577 (1995) (PME).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EWALD_NPT_KGRID_REBUILD_TOLERANCE_A",
    "build_kspace_integers",
    "ewald_npt_kgrid_cache_bin",
    "ewald_reciprocal_energy",
    "ewald_self_energy",
    "ewald_exclusion_correction",
    "default_ewald_alpha",
]

# Absolute cubic-box change (Å) that forces a host-side rebuild of the static
# reciprocal-integer grid ``n_int`` / Ewald ``alpha`` under CPT. Within a bin the
# live cell still updates every step via ``box_override``; only the integer
# k-shell set is reused (same idea as a fixed neighbor-list capacity).
EWALD_NPT_KGRID_REBUILD_TOLERANCE_A = 0.5


def ewald_npt_kgrid_cache_bin(
    box_length_A: float,
    *,
    tolerance_A: float = EWALD_NPT_KGRID_REBUILD_TOLERANCE_A,
) -> int:
    """Bin index for MM-factory cache keys under NpT (nearest tolerance bin)."""
    L = float(box_length_A)
    tol = float(tolerance_A)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(f"box_length_A must be a positive finite length, got {box_length_A!r}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tolerance_A must be a positive finite length, got {tolerance_A!r}")
    return int(np.floor(L / tol + 0.5))


def default_ewald_alpha(rcut: float, *, accuracy_exponent: float = 3.5) -> float:
    """Ewald splitting parameter so ``erfc(alpha * rcut) ~= 10^-accuracy_exponent``.

    ``accuracy_exponent=3.5`` matches ``erfc(x) < 1e-6`` at ``x ~= 3.5`` -- the
    real-space sum truncated at ``rcut`` then drops the remaining tail cleanly
    with no CHARMM-style switching function needed (unlike plain 1/r Coulomb).
    """
    return float(accuracy_exponent) / float(rcut)


def build_kspace_integers(
    cell: np.ndarray, alpha: float, *, accuracy_exponent: float = 3.5
) -> np.ndarray:
    """Host-side (static-shape) integer reciprocal-lattice vectors ``n``.

    Returns an ``(n_k, 3)`` int array (``k = n @ B``, ``B`` the reciprocal
    cell) covering every ``k`` with ``|k| <= k_cut = 2*alpha*accuracy_exponent``
    (chosen so ``exp(-k_cut^2/(4 alpha^2)) ~= exp(-accuracy_exponent^2)``,
    matching the real-space truncation's accuracy), excluding ``n=(0,0,0)``.

    Called once at term-build time with the concrete (numpy) box -- this is
    the k-space analogue of the real-space neighbor list's fixed padded shape.
    Under NPT the *integers* stay fixed for the life of the term; only the
    cartesian ``k`` vectors (``n @ B(cell)``) are recomputed from the current
    box each step, exactly like the real-space pair list reuses a fixed
    ``pair_i``/``pair_j`` shape with per-step MIC displacements.
    """
    cell = np.asarray(cell, dtype=np.float64)
    volume = float(np.abs(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError(f"non-positive cell volume {volume}")
    recip = 2.0 * np.pi * np.linalg.inv(cell).T  # rows = reciprocal vectors b_i
    k_cut = 2.0 * float(alpha) * float(accuracy_exponent)

    # generous per-axis bound: how many reciprocal shells until |n_i * b_i| > k_cut
    b_norms = np.linalg.norm(recip, axis=1)
    n_max = np.maximum(np.ceil(k_cut / np.maximum(b_norms, 1e-12)).astype(int), 1)

    n1 = np.arange(-n_max[0], n_max[0] + 1)
    n2 = np.arange(-n_max[1], n_max[1] + 1)
    n3 = np.arange(-n_max[2], n_max[2] + 1)
    grid = np.stack(np.meshgrid(n1, n2, n3, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[~np.all(grid == 0, axis=1)]

    k = grid @ recip
    k2 = np.sum(k * k, axis=1)
    keep = k2 <= k_cut * k_cut
    return grid[keep].astype(np.int32)


def ewald_reciprocal_energy(R, q, cell, n_int, alpha):
    """Reciprocal-space Coulomb energy (raw, no Coulomb constant).

    ``R``: ``(n_atoms, 3)`` cartesian positions. ``q``: ``(n_atoms,)`` charges.
    ``cell``: ``(3, 3)`` box (rows = lattice vectors, matches
    ``pbc_utils_jax.cart_coords``' ``r = s @ cell`` convention). ``n_int``:
    static ``(n_k, 3)`` integer reciprocal-lattice vectors from
    :func:`build_kspace_integers`. ``alpha``: Ewald splitting parameter.
    """
    import jax.numpy as jnp

    cell = jnp.asarray(cell)
    volume = jnp.abs(jnp.linalg.det(cell))
    recip = 2.0 * jnp.pi * jnp.linalg.inv(cell).T  # rows = b_i, differentiable in cell
    k = jnp.asarray(n_int, dtype=cell.dtype) @ recip  # (n_k, 3)
    k2 = jnp.sum(k * k, axis=-1)  # (n_k,)

    kr = R @ k.T  # (n_atoms, n_k)
    q_col = q[:, None]
    s_re = jnp.sum(q_col * jnp.cos(kr), axis=0)  # (n_k,)
    s_im = jnp.sum(q_col * jnp.sin(kr), axis=0)
    s2 = s_re * s_re + s_im * s_im

    weight = jnp.exp(-k2 / (4.0 * alpha * alpha)) / k2
    return (2.0 * jnp.pi / volume) * jnp.sum(weight * s2)


def ewald_self_energy(q, alpha):
    """Self-interaction correction (raw, no Coulomb constant)."""
    import jax.numpy as jnp

    return -(alpha / jnp.sqrt(jnp.pi)) * jnp.sum(q * q)


def ewald_exclusion_correction(R, q, cell, excl_i, excl_j, alpha):
    """Subtract the reciprocal sum's implicit contribution from excluded pairs.

    ``excl_i``/``excl_j``: static ``(n_excl,)`` int32 index arrays (bonded
    1-2/1-3 exclusions -- the reciprocal-space sum has no notion of exclusions
    since it sums over all atoms independently of the pair list). Empty
    arrays are fine (``jnp.sum`` of an empty axis is 0).
    """
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

    if excl_i.shape[0] == 0:
        return jnp.array(0.0, dtype=R.dtype)

    ri = R[excl_i]
    rj = R[excl_j]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, cell))(ri, rj)
    r = jnp.linalg.norm(disp, axis=-1)
    r_safe = jnp.maximum(r, 1e-10)
    qq = q[excl_i] * q[excl_j]
    return -jnp.sum(qq * jsp.erf(alpha * r_safe) / r_safe)
