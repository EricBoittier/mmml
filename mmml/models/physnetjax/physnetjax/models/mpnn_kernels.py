"""Shared numerical kernels for PhysNet-family message-passing models.

These helpers are the first harmonization layer between :class:`PhysNet` and
:class:`SpookyPhysNet`. They must stay **checkpoint-neutral**: no Flax module
names, no parameter trees — pure functions of arrays and scalar hyperparameters.

See ``docs/mpnn-tool-inventory.md``.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import e3x
import jax
import jax.numpy as jnp

Array = jnp.ndarray

# (e² / 4πε₀) / 2 in eV·Å — pair Coulomb prefactor used by PhysNet electrostatics.
COULOMB_PAIR_FACTOR_EV_A = 7.199822675975274

# Numerical floors for switch / distance kernels (not physical cutoffs).
_SWITCH_EPS = 1e-6
_MIN_DIST_A = 0.01


def pair_displacements(
    positions: Array,
    dst_idx: Array,
    src_idx: Array,
    *,
    cell: Optional[Array] = None,
    use_pbc: bool = False,
) -> Array:
    """Gather pairwise displacements; optional minimum-image convention."""
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    if use_pbc and cell is not None:
        dR = positions_src - positions_dst
        dS = jax.scipy.linalg.solve(cell.T, dR.T, assume_a="gen").T
        dS_mic = dS - jnp.round(dS)
        return dS_mic @ cell
    return positions_src - positions_dst


def radial_spherical_basis(
    displacements: Array,
    *,
    num_basis_functions: int,
    max_degree: int,
    cutoff: float,
    batch_mask: Optional[Array] = None,
    edge_mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Build e3x radial/spherical basis with padding-safe edge gating.

    Masked pairs are shifted off ``r=0`` before the basis (padding–padding pairs
    are coincident; zeroing a singular gradient otherwise yields ``0 * NaN`` in
    forces), then the basis is multiplied by the edge gate.

    Returns
    -------
    basis, displacements
        ``displacements`` are the **original** (unshifted) vectors so downstream
        electrostatic switches can apply their own masking.
    """
    edge_gate = None
    if batch_mask is not None or edge_mask is not None:
        edge_gate = jnp.ones(displacements.shape[0], dtype=displacements.dtype)
        if batch_mask is not None:
            edge_gate = edge_gate * jnp.asarray(
                batch_mask, dtype=displacements.dtype
            ).reshape(-1)
        if edge_mask is not None:
            edge_gate = edge_gate * jnp.asarray(
                edge_mask, dtype=displacements.dtype
            ).reshape(-1)

    basis_displacements = displacements
    if edge_gate is not None:
        basis_displacements = displacements + (1.0 - edge_gate.reshape(-1, 1))

    basis = e3x.nn.basis(
        basis_displacements,
        num=int(num_basis_functions),
        max_degree=int(max_degree),
        radial_fn=e3x.nn.exponential_chebyshev,
        cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=float(cutoff)),
    )
    if edge_gate is not None:
        basis = basis * edge_gate.astype(basis.dtype).reshape(
            -1, *([1] * (basis.ndim - 1))
        )
    return basis, displacements


def calc_electrostatics_switches(
    displacements: Array,
    batch_mask: Array,
    *,
    switch_start: float,
    switch_end: float,
    electrostatics_off_start: float,
    electrostatics_off_end: float,
    electrostatics_damping_sigma: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """Switched short-/long-range Coulomb distance factors ``(r, off_dist, eshift)``.

    Matches the PhysNet / SpookyPhysNet ``_calc_switches`` implementation,
    including optional Gaussian damping via ``erf(r / σ)``.
    """
    eps = _SWITCH_EPS
    min_dist = _MIN_DIST_A
    displacements = jnp.nan_to_num(displacements, nan=0.0, posinf=0.0, neginf=0.0)
    displacements = displacements + (1 - batch_mask)[..., None]
    squared_distances = jnp.sum(displacements**2, axis=1)
    distances = jnp.sqrt(jnp.maximum(squared_distances, min_dist**2))

    switch_dist = e3x.nn.smooth_switch(distances, switch_start, switch_end)
    off_dist = 1.0 - e3x.nn.smooth_switch(
        distances, electrostatics_off_start, electrostatics_off_end
    )
    switch_dist = jnp.clip(switch_dist, 0.0, 1.0)
    off_dist = jnp.clip(off_dist, 0.0, 1.0)
    one_minus_switch_dist = 1 - switch_dist

    safe_distances = distances + eps
    r1 = switch_dist / jnp.sqrt(squared_distances + 1.0)
    r2 = one_minus_switch_dist / safe_distances
    r = r1 + r2
    if electrostatics_damping_sigma > 0.0:
        sigma = jnp.asarray(electrostatics_damping_sigma, dtype=distances.dtype)
        r *= jax.scipy.special.erf(distances / sigma)
    eshift = safe_distances / (switch_end**2) - 2.0 / switch_end
    off_dist *= batch_mask
    eshift *= batch_mask
    r = jnp.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    off_dist = jnp.nan_to_num(off_dist, nan=0.0, posinf=0.0, neginf=0.0)
    eshift = jnp.nan_to_num(eshift, nan=0.0, posinf=0.0, neginf=0.0)
    return r, off_dist, eshift


def pair_electrostatics_energy(
    atomic_charges: Array,
    r: Array,
    off_dist: Array,
    eshift: Array,
    dst_idx: Array,
    src_idx: Array,
    batch_mask: Array,
    batch_segments: Array,
    batch_size: int,
) -> Tuple[Array, Array]:
    """Assemble pair Coulomb energy and reduce to atomic / batch totals.

    Returns ``(atomic_electrostatics, batch_electrostatics)`` with trailing
    singleton axes ``(..., 1, 1, 1)`` matching PhysNet energy layouts.
    """
    q1 = jnp.clip(jnp.take(atomic_charges, dst_idx, fill_value=0.0), -10.0, 10.0)
    q2 = jnp.clip(jnp.take(atomic_charges, src_idx, fill_value=0.0), -10.0, 10.0)
    electrostatics = COULOMB_PAIR_FACTOR_EV_A * q1 * q2 * (r + eshift) * batch_mask
    electrostatics *= off_dist
    num_atoms_actual = atomic_charges.shape[0]
    atomic_electrostatics = jax.ops.segment_sum(
        electrostatics,
        segment_ids=dst_idx,
        num_segments=num_atoms_actual,
    )
    atomic_electrostatics = atomic_electrostatics[:num_atoms_actual]
    batch_electrostatics = jax.ops.segment_sum(
        atomic_electrostatics,
        segment_ids=batch_segments,
        num_segments=batch_size,
    )
    atomic_electrostatics = atomic_electrostatics[..., None, None, None]
    batch_electrostatics = batch_electrostatics[..., None, None, None]
    return atomic_electrostatics, batch_electrostatics


__all__ = [
    "COULOMB_PAIR_FACTOR_EV_A",
    "calc_electrostatics_switches",
    "pair_displacements",
    "pair_electrostatics_energy",
    "radial_spherical_basis",
]
