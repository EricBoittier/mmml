"""Prototype direct multipolar electrostatics for a future FMM backend.

This module is deliberately small and Cartesian.  It is not a production
long-range solver; it provides executable reference behavior for the extension
sketched in ``pasted-text.txt``: source multipoles, point target potentials,
target multipole contractions, self masking, and JAX-differentiable energies.

The eventual jaxFMM integration should convert these Cartesian moments into
jaxFMM's real solid-harmonic coefficient convention before P2M/M2M/M2L/L2L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

MultipoleOrder = Literal[0, 1, 2]

_QUAD_COMPONENTS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)


@dataclass(frozen=True)
class CartesianMultipoleLayout:
    """Packed Cartesian multipole coefficient layout through quadrupoles."""

    order: MultipoleOrder

    @property
    def ncoeff(self) -> int:
        return multipole_coeff_count(self.order)

    @property
    def charge_index(self) -> int:
        return 0

    @property
    def dipole_slice(self) -> slice:
        return slice(1, 4)

    @property
    def quadrupole_slice(self) -> slice:
        return slice(4, 10)


def multipole_coeff_count(order: MultipoleOrder) -> int:
    """Return packed coefficient count for Cartesian ranks 0..``order``."""
    if order == 0:
        return 1
    if order == 1:
        return 4
    if order == 2:
        return 10
    raise ValueError(f"unsupported multipole order {order!r}; expected 0, 1, or 2")


def pack_cartesian_multipoles(
    charge: jax.Array,
    *,
    dipole: jax.Array | None = None,
    quadrupole: jax.Array | None = None,
    order: MultipoleOrder = 0,
) -> jax.Array:
    """Pack charge, dipole, and quadrupole moments into one coefficient array.

    The quadrupole is stored as the six independent symmetric tensor entries
    ``xx, xy, xz, yy, yz, zz``.  No traceless projection is applied here because
    different FMM and force-field conventions make different choices.
    """
    charge = jnp.asarray(charge)
    coeffs = jnp.zeros(
        charge.shape + (multipole_coeff_count(order),),
        dtype=charge.dtype,
    )
    coeffs = coeffs.at[..., 0].set(charge)
    if order >= 1:
        if dipole is None:
            dipole = jnp.zeros(charge.shape + (3,), dtype=charge.dtype)
        coeffs = coeffs.at[..., 1:4].set(jnp.asarray(dipole, dtype=charge.dtype))
    if order >= 2:
        if quadrupole is None:
            quadrupole = jnp.zeros(charge.shape + (3, 3), dtype=charge.dtype)
        quad = jnp.asarray(quadrupole, dtype=charge.dtype)
        packed_quad = jnp.stack(
            [quad[..., a, b] for a, b in _QUAD_COMPONENTS],
            axis=-1,
        )
        coeffs = coeffs.at[..., 4:10].set(packed_quad)
    return coeffs


def unpack_cartesian_multipoles(
    coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unpack coefficients into charge, dipole, and symmetric quadrupole tensor."""
    coeffs = jnp.asarray(coeffs)
    if coeffs.shape[-1] not in (1, 4, 10):
        raise ValueError("expected packed Cartesian multipoles with 1, 4, or 10 coeffs")
    charge = coeffs[..., 0]
    dipole = jnp.zeros(coeffs.shape[:-1] + (3,), dtype=coeffs.dtype)
    quadrupole = jnp.zeros(coeffs.shape[:-1] + (3, 3), dtype=coeffs.dtype)
    if coeffs.shape[-1] >= 4:
        dipole = coeffs[..., 1:4]
    if coeffs.shape[-1] >= 10:
        for offset, (a, b) in enumerate(_QUAD_COMPONENTS):
            value = coeffs[..., 4 + offset]
            quadrupole = quadrupole.at[..., a, b].set(value)
            if a != b:
                quadrupole = quadrupole.at[..., b, a].set(value)
    return charge, dipole, quadrupole


def pair_potential_from_cartesian_multipole(
    displacement: jax.Array,
    source_coeffs: jax.Array,
    *,
    softening: float = 0.0,
) -> jax.Array:
    """Potential at a target from one Cartesian source multipole.

    ``displacement`` is ``target_position - source_position``.  Through
    quadrupoles, the convention is:

    ``q/r + mu_a r_a/r^3 + 0.5 Q_ab (3 r_a r_b - delta_ab r^2)/r^5``.
    """
    displacement = jnp.asarray(displacement)
    source_coeffs = jnp.asarray(source_coeffs)
    charge, dipole, quadrupole = unpack_cartesian_multipoles(source_coeffs)
    r2 = jnp.dot(displacement, displacement) + softening * softening
    nonzero = r2 > 0.0
    safe_r2 = jnp.where(nonzero, r2, 1.0)
    inv_r = jnp.where(nonzero, jax.lax.rsqrt(safe_r2), 0.0)
    inv_r3 = inv_r**3
    inv_r5 = inv_r**5
    potential = charge * inv_r
    if source_coeffs.shape[-1] >= 4:
        potential = potential + jnp.dot(dipole, displacement) * inv_r3
    if source_coeffs.shape[-1] >= 10:
        rr = jnp.outer(displacement, displacement)
        hess_green = (
            3.0 * rr - jnp.eye(3, dtype=displacement.dtype) * r2
        ) * inv_r5
        potential = potential + 0.5 * jnp.sum(quadrupole * hess_green)
    return potential


def direct_multipole_to_point(
    source_positions: jax.Array,
    source_coeffs: jax.Array,
    target_positions: jax.Array,
    *,
    exclude_self: bool = False,
    softening: float = 0.0,
) -> jax.Array:
    """Evaluate source multipole potentials at point targets by direct summation."""
    source_positions = jnp.asarray(source_positions)
    source_coeffs = jnp.asarray(source_coeffs)
    target_positions = jnp.asarray(target_positions)

    def target_potential(target_position: jax.Array, target_index: jax.Array) -> jax.Array:
        displacement = target_position[None, :] - source_positions
        pair_phi = jax.vmap(
            lambda dr, coeff: pair_potential_from_cartesian_multipole(
                dr,
                coeff,
                softening=softening,
            )
        )(displacement, source_coeffs)
        if exclude_self:
            pair_phi = jnp.where(
                jnp.arange(source_positions.shape[0]) == target_index,
                0.0,
                pair_phi,
            )
        return jnp.sum(pair_phi)

    return jax.vmap(target_potential)(target_positions, jnp.arange(target_positions.shape[0]))


def direct_multipole_to_multipole_energy(
    source_positions: jax.Array,
    source_coeffs: jax.Array,
    target_positions: jax.Array,
    target_coeffs: jax.Array,
    *,
    exclude_self: bool = False,
    softening: float = 0.0,
) -> jax.Array:
    """Return per-target energy from source multipoles and target moments."""
    source_positions = jnp.asarray(source_positions)
    source_coeffs = jnp.asarray(source_coeffs)
    target_positions = jnp.asarray(target_positions)
    target_coeffs = jnp.asarray(target_coeffs)

    def source_potential_at(target_position: jax.Array) -> jax.Array:
        return direct_multipole_to_point(
            source_positions,
            source_coeffs,
            target_position[None, :],
            softening=softening,
        )[0]

    grad_fn = jax.grad(source_potential_at)
    hess_fn = jax.hessian(source_potential_at)

    def target_energy(
        target_position: jax.Array,
        target_coeff: jax.Array,
        target_index: jax.Array,
    ) -> jax.Array:
        charge, dipole, quadrupole = unpack_cartesian_multipoles(target_coeff)
        if exclude_self:
            masked_coeffs = jnp.where(
                jnp.arange(source_positions.shape[0])[:, None] == target_index,
                0.0,
                source_coeffs,
            )
            phi = direct_multipole_to_point(
                source_positions,
                masked_coeffs,
                target_position[None, :],
                softening=softening,
            )[0]

            def masked_source_potential_at(position: jax.Array) -> jax.Array:
                return direct_multipole_to_point(
                    source_positions,
                    masked_coeffs,
                    position[None, :],
                    softening=softening,
                )[0]

            grad_phi = jax.grad(masked_source_potential_at)(target_position)
            hess_phi = jax.hessian(masked_source_potential_at)(target_position)
        else:
            phi = source_potential_at(target_position)
            grad_phi = grad_fn(target_position)
            hess_phi = hess_fn(target_position)
        energy = charge * phi
        if target_coeff.shape[-1] >= 4:
            energy = energy + jnp.dot(dipole, grad_phi)
        if target_coeff.shape[-1] >= 10:
            energy = energy + 0.5 * jnp.sum(quadrupole * hess_phi)
        return energy

    return jax.vmap(target_energy)(
        target_positions,
        target_coeffs,
        jnp.arange(target_positions.shape[0]),
    )


def self_energy(
    positions: jax.Array,
    coeffs: jax.Array,
    *,
    softening: float = 0.0,
) -> jax.Array:
    """Total pair energy for one shared source-target multipole set."""
    per_target = direct_multipole_to_multipole_energy(
        positions,
        coeffs,
        positions,
        coeffs,
        exclude_self=True,
        softening=softening,
    )
    return 0.5 * jnp.sum(per_target)
