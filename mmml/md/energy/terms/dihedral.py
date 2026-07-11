"""Dihedral-angle harmonic restraint term (φ/ψ backbone restraints).

Extracted from ``examples/cg_jaxmd.py`` (``_dihedral_angle_rad``,
``_periodic_angle_delta_rad``, ``_phi_psi_restraint_energy``), with the module
globals (``PHI_CENTRAL``, ``PSI_CENTRAL``, targets, force constant) lifted into
constructor arguments so the term is reusable across systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax
from mmml.md.system import MolecularSystem

__all__ = ["DihedralRestraint", "DihedralRestraintTerm"]


@dataclass(frozen=True)
class DihedralRestraint:
    """One harmonic restraint ``0.5 k (Δθ)²`` on a signed dihedral.

    ``indices`` are the four atom indices defining the dihedral, ``target_deg``
    the restraint center, and ``k_ev`` the force constant (eV / rad²).
    """

    indices: tuple[int, int, int, int]
    target_deg: float
    k_ev: float


def _dihedral_angle_rad(r, atom_indices):
    """Signed dihedral angle in radians for four atom indices."""
    import jax.numpy as jnp

    p0 = r[atom_indices[0]]
    p1 = r[atom_indices[1]]
    p2 = r[atom_indices[2]]
    p3 = r[atom_indices[3]]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / jnp.linalg.norm(b1)
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.arctan2(y, x)


def _periodic_angle_delta_rad(angle, target):
    """Smallest signed angle difference ``angle - target`` in radians."""
    import jax.numpy as jnp

    return jnp.arctan2(jnp.sin(angle - target), jnp.cos(angle - target))


@register_term("dihedral")
class DihedralRestraintTerm:
    """Composable :class:`~mmml.md.energy.registry.EnergyTerm` of dihedral restraints."""

    name = "dihedral"

    def __init__(self, restraints: Sequence[DihedralRestraint]):
        self.restraints = list(restraints)

    def neighbor_request(self, system: MolecularSystem):
        return None  # geometry-only, no pair list

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        specs = [
            (
                jnp.asarray(r.indices, dtype=jnp.int32),
                jnp.deg2rad(float(r.target_deg)),
                float(r.k_ev),
            )
            for r in self.restraints
        ]

        def energy_fn(R, **kwargs) -> Any:
            total = jnp.asarray(0.0)
            for idx, target, k in specs:
                angle = _dihedral_angle_rad(R, idx)
                delta = _periodic_angle_delta_rad(angle, target)
                total = total + 0.5 * k * delta * delta
            return total

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )
