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
QuadrupoleTracePolicy = Literal["traceless", "preserve"]

E3X_CONVENTION_NOTE = (
    "Use e3x.Config to pin global irrep conventions before converting Cartesian "
    "moments with e3x.so3.tensor_to_irreps or e3x.so3.irreps_to_tensor. The "
    "prototype boundary uses symmetric traceless degree-2 quadrupoles because "
    "that is the Cartesian tensor represented by an e3x degree-2 irrep."
)

SR_ML_FMM_COMPOSITION_NOTE = (
    "When a short-range ML potential has implicit electrostatics, do not add a "
    "full explicit FMM energy for the same owned pairs. Either train the ML "
    "model as a short-range residual after subtracting explicit electrostatics, "
    "or add only a smooth long-range FMM complement outside the ML training "
    "cutoff. MM/MM can own its full explicit multipolar FMM term; MM->ML should "
    "be one-way embedding unless reciprocal MM/ML energy ownership is defined."
)

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


@dataclass(frozen=True)
class MmMmMultipolarResult:
    """Owned MM/MM multipolar energy with self pairs excluded and pairs halved."""

    energy: jax.Array
    per_site_energy: jax.Array


@dataclass(frozen=True)
class MmToMlMultipolarResult:
    """One-way MM->ML embedding quantities without pair halving."""

    potential: jax.Array
    potential_gradient: jax.Array
    electric_field: jax.Array
    target_energy: jax.Array | None = None


def multipole_coeff_count(order: MultipoleOrder) -> int:
    """Return packed coefficient count for Cartesian ranks 0..``order``."""
    if order == 0:
        return 1
    if order == 1:
        return 4
    if order == 2:
        return 10
    raise ValueError(f"unsupported multipole order {order!r}; expected 0, 1, or 2")


def symmetric_traceless(tensor: jax.Array) -> jax.Array:
    """Return the symmetric traceless degree-2 Cartesian tensor.

    E3x degree-2 irreps correspond to symmetric traceless Cartesian tensors, so
    this is the default quadrupole boundary convention used by this prototype.
    """
    tensor = jnp.asarray(tensor)
    symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
    trace = jnp.trace(symmetric, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=symmetric.dtype)
    return symmetric - trace[..., None, None] * identity / 3.0


def pack_cartesian_multipoles(
    charge: jax.Array,
    *,
    dipole: jax.Array | None = None,
    quadrupole: jax.Array | None = None,
    order: MultipoleOrder = 0,
    quadrupole_trace_policy: QuadrupoleTracePolicy = "traceless",
) -> jax.Array:
    """Pack charge, dipole, and quadrupole moments into one coefficient array.

    The quadrupole is stored as the six independent symmetric tensor entries
    ``xx, xy, xz, yy, yz, zz``.  By default, it is projected to the symmetric
    traceless degree-2 Cartesian tensor convention used by E3x irreps.  Use
    ``quadrupole_trace_policy="preserve"`` only for legacy Cartesian tests.
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
        if quadrupole_trace_policy == "traceless":
            quad = symmetric_traceless(quad)
        elif quadrupole_trace_policy != "preserve":
            raise ValueError(
                "quadrupole_trace_policy must be 'traceless' or 'preserve'"
            )
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


def direct_multipole_potential_gradient_to_point(
    source_positions: jax.Array,
    source_coeffs: jax.Array,
    target_position: jax.Array,
    *,
    softening: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Return potential and gradient of potential at one target point."""

    def potential_at(position: jax.Array) -> jax.Array:
        return direct_multipole_to_point(
            source_positions,
            source_coeffs,
            position[None, :],
            softening=softening,
        )[0]

    return potential_at(target_position), jax.grad(potential_at)(target_position)


def direct_multipole_potential_gradient_to_points(
    source_positions: jax.Array,
    source_coeffs: jax.Array,
    target_positions: jax.Array,
    *,
    softening: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Return potentials and potential gradients at target points."""
    return jax.vmap(
        lambda target_position: direct_multipole_potential_gradient_to_point(
            source_positions,
            source_coeffs,
            target_position,
            softening=softening,
        )
    )(target_positions)


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


def mm_mm_multipolar_energy(
    mm_positions: jax.Array,
    mm_coeffs: jax.Array,
    *,
    softening: float = 0.0,
) -> MmMmMultipolarResult:
    """MM/MM owned energy: exclude self terms and halve pair contributions."""
    per_site_energy = direct_multipole_to_multipole_energy(
        mm_positions,
        mm_coeffs,
        mm_positions,
        mm_coeffs,
        exclude_self=True,
        softening=softening,
    )
    return MmMmMultipolarResult(
        energy=0.5 * jnp.sum(per_site_energy),
        per_site_energy=per_site_energy,
    )


def mm_to_ml_multipolar_embedding(
    mm_positions: jax.Array,
    mm_coeffs: jax.Array,
    ml_positions: jax.Array,
    ml_target_coeffs: jax.Array | None = None,
    *,
    softening: float = 0.0,
) -> MmToMlMultipolarResult:
    """One-way MM->ML embedding potential/field with no pair halving.

    If ``ml_target_coeffs`` is supplied, target energies are contracted against
    the MM-generated potential derivatives.  This path is appropriate when the
    ML term consumes an external MM field and does not also own the reciprocal
    ML->MM or MM/ML pair energy separately.
    """
    potential, potential_gradient = direct_multipole_potential_gradient_to_points(
        mm_positions,
        mm_coeffs,
        ml_positions,
        softening=softening,
    )
    electric_field = -potential_gradient
    target_energy = None
    if ml_target_coeffs is not None:
        target_energy = direct_multipole_to_multipole_energy(
            mm_positions,
            mm_coeffs,
            ml_positions,
            ml_target_coeffs,
            softening=softening,
        )
    return MmToMlMultipolarResult(
        potential=potential,
        potential_gradient=potential_gradient,
        electric_field=electric_field,
        target_energy=target_energy,
    )
