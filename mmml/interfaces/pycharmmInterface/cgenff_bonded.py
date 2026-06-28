"""JAX CGENFF bonded energy and forces (bond/angle/torsion/improper).

Formulas follow :mod:`jax_md.mm_forcefields.oplsaa.energy` so MMML bonded terms
can be cross-checked against the jax-md reference implementation.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, vmap

from jax_md import space
from jax_md.mm_forcefields.base import BondedParameters, Topology
from jax_md.util import normalize, safe_arccos, safe_norm

KCAL_MOL_TO_EV = 0.04336411530877155


def free_space_displacement() -> space.DisplacementFn:
    disp_fn, _ = space.free()
    return disp_fn


def bonded_energy_components(
    positions: Array,
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
) -> dict[str, Array]:
    """Return bonded energy components in kcal/mol (jax-md convention)."""
    if displacement_fn is None:
        displacement_fn = free_space_displacement()

    def bond_energy() -> Array:
        if topology.bonds.shape[0] == 0:
            return jnp.array(0.0, dtype=positions.dtype)
        i, j = topology.bonds[:, 0], topology.bonds[:, 1]
        disp = vmap(displacement_fn)(positions[i], positions[j])
        r = safe_norm(disp)
        return jnp.sum(bonded.bond_k * (r - bonded.bond_r0) ** 2)

    def angle_energy() -> Array:
        if topology.angles.shape[0] == 0:
            return jnp.array(0.0, dtype=positions.dtype)
        i, j, k = (
            topology.angles[:, 0],
            topology.angles[:, 1],
            topology.angles[:, 2],
        )
        rij = vmap(displacement_fn)(positions[i], positions[j])
        rkj = vmap(displacement_fn)(positions[k], positions[j])
        rij_norm = normalize(rij)
        rkj_norm = normalize(rkj)
        cos_theta = jnp.sum(rij_norm * rkj_norm, axis=-1)
        theta = safe_arccos(cos_theta)
        return jnp.sum(bonded.angle_k * (theta - bonded.angle_theta0) ** 2)

    def torsion_energy() -> Array:
        if topology.torsions.shape[0] == 0:
            return jnp.array(0.0, dtype=positions.dtype)

        idx = topology.torsions

        def compute_dihedral(p0, p1, p2, p3):
            b0 = displacement_fn(p1, p0)
            b1 = displacement_fn(p2, p1)
            b2 = displacement_fn(p3, p2)
            n1 = normalize(jnp.cross(b0, b1))
            n2 = normalize(jnp.cross(b1, b2))
            cos_phi = jnp.sum(n1 * n2)
            return safe_arccos(cos_phi)

        phi = vmap(compute_dihedral)(
            positions[idx[:, 0]],
            positions[idx[:, 1]],
            positions[idx[:, 2]],
            positions[idx[:, 3]],
        )
        return jnp.sum(
            bonded.torsion_k
            * (1 + jnp.cos(bonded.torsion_n * phi - bonded.torsion_gamma))
        )

    def improper_energy() -> Array:
        if topology.impropers.shape[0] == 0:
            return jnp.array(0.0, dtype=positions.dtype)

        idx = topology.impropers
        improper_n = jnp.asarray(bonded.improper_n)

        def compute_dihedral_signed(p0, p1, p2, p3):
            b0 = displacement_fn(p1, p0)
            b1 = displacement_fn(p2, p1)
            b2 = displacement_fn(p3, p2)
            b1_norm = normalize(b1)
            v = b0 - jnp.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
            w = b2 - jnp.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm
            x = jnp.sum(v * w, axis=-1)
            y = jnp.sum(jnp.cross(b1_norm, v) * w, axis=-1)
            return jnp.arctan2(y, x)

        psi = vmap(compute_dihedral_signed)(
            positions[idx[:, 0]],
            positions[idx[:, 1]],
            positions[idx[:, 2]],
            positions[idx[:, 3]],
        )
        return jnp.sum(
            bonded.improper_k
            * (1 + jnp.cos(improper_n * psi - bonded.improper_gamma))
        )

    e_bond = bond_energy()
    e_angle = angle_energy()
    e_torsion = torsion_energy()
    e_improper = improper_energy()
    e_total = e_bond + e_angle + e_torsion + e_improper
    return {
        "bond": e_bond,
        "angle": e_angle,
        "torsion": e_torsion,
        "improper": e_improper,
        "total": e_total,
    }


def bonded_energy_and_forces(
    positions: Array,
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    energy_unit: str = "kcal/mol",
) -> tuple[dict[str, Array], Array]:
    """Bonded energy (dict) and forces (N, 3) for a CGENFF bonded model."""
    if displacement_fn is None:
        displacement_fn = free_space_displacement()

    def total_energy(pos: Array) -> Array:
        return bonded_energy_components(pos, topology, bonded, displacement_fn)["total"]

    components = bonded_energy_components(positions, topology, bonded, displacement_fn)
    forces = -jax.grad(total_energy)(positions)

    scale = 1.0
    if energy_unit == "eV":
        scale = KCAL_MOL_TO_EV
        components = {k: v * scale for k, v in components.items()}
        forces = forces * scale
    elif energy_unit != "kcal/mol":
        raise ValueError(f"Unsupported energy_unit: {energy_unit!r}")

    return components, forces


def build_bonded_energy_fn(
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    energy_unit: str = "kcal/mol",
) -> Callable[[Array], tuple[dict[str, Array], Array]]:
    """Return ``(positions) -> (components, forces)`` for reuse in calculators."""

    def evaluate(positions: Array) -> tuple[dict[str, Array], Array]:
        return bonded_energy_and_forces(
            positions,
            topology,
            bonded,
            displacement_fn,
            energy_unit=energy_unit,
        )

    return evaluate
