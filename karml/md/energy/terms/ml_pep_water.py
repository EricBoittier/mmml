"""ML core–group intermolecular energy term (e.g. peptide–water dimers).

Thin wrapper over
``jaxmdInterface.hybrid_energy.make_peptide_water_ml_energy_fn``: the trained
model evaluates each core+molecule dimer, subtracting the redundant core and
molecule monomer energies to yield
``E_core + Σ E_group + Σ E_core-group(ML)``. Dimers are evaluated over a padded
set of active groups (``active_group_slots`` / ``active_group_mask``), so the set
of nearby molecules is dynamic while the graph shape stays static (decision B).

Generalized from the cg_jaxmd peptide/water naming: ``core_indices`` are the
solute atoms, ``group_indices`` the ``(n_groups, atoms_per_group)`` solvent
molecules. Model + params come from the run ``EnergyContext`` unless overridden.
"""

from __future__ import annotations

from typing import Any, Sequence

from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.energy.terms._common import resolve_displacement_fn, resolve_ml_model
from mmml.md.system import MolecularSystem

__all__ = ["MLCoreGroupTerm"]


@register_term("ml_pep_water")
class MLCoreGroupTerm:
    """ML core-monomer + core–group interaction energy over padded group slots."""

    name = "ml_pep_water"

    def __init__(
        self,
        core_indices: Sequence[int],
        group_indices: Sequence[Sequence[int]],
        model: Any = None,
        params: Any = None,
        charges: Any = None,
        spins: Any = None,
        interaction_cutoff_A: float | None = None,
        interaction_switch_width_A: float = 2.0,
    ):
        self.core_indices = core_indices
        self.group_indices = group_indices
        self.model = model
        self.params = params
        self.charges = charges
        self.spins = spins
        self.interaction_cutoff_A = interaction_cutoff_A
        self.interaction_switch_width_A = interaction_switch_width_A

    def neighbor_request(self, system: MolecularSystem):
        if self.interaction_cutoff_A is None:
            return None
        return NeighborRequest(cutoff_A=float(self.interaction_cutoff_A), kind="ml_core_group")

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        from mmml.interfaces.jaxmdInterface.hybrid_energy import (
            make_peptide_water_ml_energy_fn,
        )

        model, params = resolve_ml_model(self, ctx)
        displacement_fn = resolve_displacement_fn(system, ctx)
        jax_z = jnp.asarray(system.Z, dtype=jnp.int32)
        core_idx = jnp.asarray(self.core_indices, dtype=jnp.int32)
        group_idx = [jnp.asarray(g, dtype=jnp.int32) for g in self.group_indices]

        dimer_energy_fn = make_peptide_water_ml_energy_fn(
            model, params, jax_z, core_idx, group_idx, displacement_fn,
            charges=self.charges, spins=self.spins,
            interaction_cutoff_A=self.interaction_cutoff_A,
            interaction_switch_width_A=self.interaction_switch_width_A,
        )

        def energy_fn(R, *, active_group_slots=None, active_group_mask=None, **kwargs) -> Any:
            return dimer_energy_fn(
                R,
                active_water_slots=active_group_slots,
                active_water_mask=active_group_mask,
            )

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=None,
            neighbor_request=self.neighbor_request(system),
        )
