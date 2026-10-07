"""JAX CGENFF bonded energy and forces (bond/angle/torsion/improper/urey-bradley).

Formulas follow :mod:`jax_md.mm_forcefields.oplsaa.energy` so MMML bonded terms
can be cross-checked against the jax-md reference implementation.  Urey–Bradley
1–3 distance terms (from CHARMM angle lines with ``K_ub`` / ``S0``) are evaluated
alongside angle terms using the same ``topology.angles`` index rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import Array, vmap

from jax_md import space
from jax_md.mm_forcefields.base import BondedParameters, Topology
from jax_md.util import normalize, safe_arccos, safe_norm

from mmml.interfaces.pycharmmInterface.cgenff_cmap import cmap_energy

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.cgenff_topology import CgenffBondedSystem

# Re-exported for callers that import it from here; the value is owned by
# mmml.data.units. This module previously carried its own literal
# (0.04336411530877155, i.e. 1 eV = 23.060541945 kcal/mol), which disagreed with
# mmml.data.units in the 8th significant figure and made two sources of truth for
# one constant. mmml.data.units is the CHARMM-consistent choice: CHARMM's own
# TOKCAL (627.5095, consta_ltm.F90) over Hartree->eV gives 23.060548992
# kcal/mol/eV, which mmml.data.units matches to 8e-09 while the old literal was
# 7e-06 off -- and it is also the closer of the two to ASE's kcal/mol.
from mmml.data.units import KCAL_MOL_TO_EV


def free_space_displacement() -> space.DisplacementFn:
    disp_fn, _ = space.free()
    return disp_fn


def urey_bradley_energy(
    positions: Array,
    topology: Topology,
    urey_k: Array | None,
    urey_r0: Array | None,
    displacement_fn: space.DisplacementFn,
) -> Array:
    """Urey–Bradley 1–3 distance energy for each angle row (kcal/mol)."""
    if urey_k is None or urey_r0 is None or topology.angles.shape[0] == 0:
        return jnp.array(0.0, dtype=positions.dtype)
    i, _, k = topology.angles[:, 0], topology.angles[:, 1], topology.angles[:, 2]
    disp = vmap(displacement_fn)(positions[i], positions[k])
    r = safe_norm(disp)
    return jnp.sum(urey_k * (r - urey_r0) ** 2)


def bonded_energy_components(
    positions: Array,
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    urey_k: Array | None = None,
    urey_r0: Array | None = None,
    include_cmap: bool = True,
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

        # CHARMM PSF impropers are I J K L (I = central atom).  Use PSF atom order
        # for the signed improper angle; n=0 terms use cos(psi-gamma), not cos(n*psi).
        psi = vmap(compute_dihedral_signed)(
            positions[idx[:, 0]],
            positions[idx[:, 1]],
            positions[idx[:, 2]],
            positions[idx[:, 3]],
        )
        # n=0 impropers are harmonic in CHARMM: E = k*(psi - psi0)^2.  The cosine
        # form 2k(1+cos(psi-gamma)) only matches to O(delta^2); its minimum sits at
        # psi-gamma = pi, so the harmonic deviation is wrap(psi - gamma - pi).
        two_pi = 2.0 * jnp.pi
        delta = jnp.mod(psi - bonded.improper_gamma, two_pi) - jnp.pi
        e_harmonic = bonded.improper_k * delta * delta
        phase = improper_n * psi - bonded.improper_gamma
        e_periodic = bonded.improper_k * (1.0 + jnp.cos(phase))
        return jnp.sum(jnp.where(improper_n == 0, e_harmonic, e_periodic))

    e_bond = bond_energy()
    e_angle = angle_energy()
    e_urey = urey_bradley_energy(
        positions, topology, urey_k, urey_r0, displacement_fn
    )
    e_torsion = torsion_energy()
    e_improper = improper_energy()
    e_cmap = (
        cmap_energy(positions, topology, bonded, displacement_fn)
        if include_cmap
        else jnp.array(0.0, dtype=positions.dtype)
    )
    e_total = e_bond + e_angle + e_urey + e_torsion + e_improper + e_cmap
    return {
        "bond": e_bond,
        "angle": e_angle,
        "urey": e_urey,
        "torsion": e_torsion,
        "improper": e_improper,
        "cmap": e_cmap,
        "total": e_total,
    }


def bonded_energy_and_forces(
    positions: Array,
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    urey_k: Array | None = None,
    urey_r0: Array | None = None,
    energy_unit: str = "kcal/mol",
    include_cmap: bool = True,
) -> tuple[dict[str, Array], Array]:
    """Bonded energy (dict) and forces (N, 3) for a CGENFF bonded model."""
    if displacement_fn is None:
        displacement_fn = free_space_displacement()

    def total_energy(pos: Array) -> Array:
        return bonded_energy_components(
            pos,
            topology,
            bonded,
            displacement_fn,
            urey_k=urey_k,
            urey_r0=urey_r0,
            include_cmap=include_cmap,
        )["total"]

    components = bonded_energy_components(
        positions,
        topology,
        bonded,
        displacement_fn,
        urey_k=urey_k,
        urey_r0=urey_r0,
        include_cmap=include_cmap,
    )
    forces = -jax.grad(total_energy)(positions)

    scale = 1.0
    if energy_unit == "eV":
        scale = KCAL_MOL_TO_EV
        components = {k: v * scale for k, v in components.items()}
        forces = forces * scale
    elif energy_unit != "kcal/mol":
        raise ValueError(f"Unsupported energy_unit: {energy_unit!r}")

    return components, forces


def bonded_energy_components_from_system(
    system: CgenffBondedSystem,
    positions: Array | None = None,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    include_cmap: bool = True,
) -> dict[str, Array]:
    """Evaluate bonded components using ``system`` Urey–Bradley arrays."""
    pos = system.positions if positions is None else positions
    return bonded_energy_components(
        pos,
        system.topology,
        system.bonded,
        displacement_fn,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
        include_cmap=include_cmap,
    )


def bonded_energy_and_forces_from_system(
    system: CgenffBondedSystem,
    positions: Array | None = None,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    energy_unit: str = "kcal/mol",
    include_cmap: bool = True,
) -> tuple[dict[str, Array], Array]:
    """Evaluate bonded energy/forces using ``system`` Urey–Bradley arrays."""
    pos = system.positions if positions is None else positions
    return bonded_energy_and_forces(
        pos,
        system.topology,
        system.bonded,
        displacement_fn,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
        energy_unit=energy_unit,
        include_cmap=include_cmap,
    )


def build_bonded_energy_fn(
    topology: Topology,
    bonded: BondedParameters,
    displacement_fn: space.DisplacementFn | None = None,
    *,
    urey_k: Array | None = None,
    urey_r0: Array | None = None,
    energy_unit: str = "kcal/mol",
) -> Callable[[Array], tuple[dict[str, Array], Array]]:
    """Return ``(positions) -> (components, forces)`` for reuse in calculators."""

    def evaluate(positions: Array) -> tuple[dict[str, Array], Array]:
        return bonded_energy_and_forces(
            positions,
            topology,
            bonded,
            displacement_fn,
            urey_k=urey_k,
            urey_r0=urey_r0,
            energy_unit=energy_unit,
        )

    return evaluate
