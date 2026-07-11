"""Steered-MD moving harmonic restraint term.

Extracted from ``examples/cg_jaxmd.py`` (``end_to_end_distance``,
``smd_bias_energy``). The collective variable is the minimum-image end-to-end
distance between two anchor atoms; the bias is a harmonic well whose center may
move each step via a ``lambda_t`` kwarg (steered MD) or stay fixed.

Globals lifted to constructor args: ``SMD_ATOM_I``/``SMD_ATOM_J`` ->
``atom_i``/``atom_j``, ``SMD_K_EV_A2`` -> ``k_ev_per_A2``. The periodic cell is
read from :attr:`MolecularSystem.box` (free space when ``box is None``).
"""

from __future__ import annotations

from typing import Any

from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax
from mmml.md.system import MolecularSystem

__all__ = ["SMDBiasTerm"]


@register_term("smd")
class SMDBiasTerm:
    """Moving harmonic restraint ``0.5 k (d(R) - λ_t)²`` on an end-to-end CV.

    ``λ_t`` is supplied per call via the ``lambda_t`` kwarg (steered MD). When
    absent it falls back to ``target`` (a fixed umbrella), defaulting to the CV
    value at build time if ``target`` is None.
    """

    name = "smd"

    def __init__(
        self,
        atom_i: int,
        atom_j: int,
        k_ev_per_A2: float,
        target: float | None = None,
    ):
        self.atom_i = int(atom_i)
        self.atom_j = int(atom_j)
        self.k_ev_per_A2 = float(k_ev_per_A2)
        self.target = target

    def neighbor_request(self, system: MolecularSystem):
        return None  # single-pair CV, no neighbor list

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        i, j = self.atom_i, self.atom_j
        k = self.k_ev_per_A2
        cell = None if system.box is None else jnp.asarray(system.box)

        def distance(R):
            if cell is None:
                disp = R[j] - R[i]
            else:
                from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
                    mic_displacement,
                )

                disp = mic_displacement(R[i], R[j], cell)
            return jnp.sqrt(jnp.sum(disp * disp) + 1e-12)

        fixed_target = self.target
        if fixed_target is None:
            fixed_target = float(distance(jnp.asarray(system.R)))

        def energy_fn(R, *, lambda_t: Any = None, **kwargs) -> Any:
            target = fixed_target if lambda_t is None else lambda_t
            d = distance(R)
            return 0.5 * k * jnp.square(d - target)

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )
