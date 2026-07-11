"""ML intramolecular monomer energy term.

Thin wrapper over ``jaxmdInterface.hybrid_energy.make_monomer_energy_fn``: the
trained model evaluates each monomer (peptide, water, …) in isolation, batched by
size with jax.vmap and PBC-unfolded via the displacement function. The energy is
in the model's native units (eV, matching ``cg_jaxmd``); no conversion applied.

Model + params come from the run :class:`~mmml.md.energy.registry.EnergyContext`
(``ctx.model`` / ``ctx.params``) unless overridden in the constructor, so the
term stays model-agnostic (physnet / spooky both work — the factory probes for
``charges`` / ``spins`` support).
"""

from __future__ import annotations

from typing import Any, Sequence

from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import resolve_displacement_fn, resolve_ml_model
from mmml.md.system import MolecularSystem

__all__ = ["MLIntramolecularTerm"]


@register_term("ml_intra")
class MLIntramolecularTerm:
    """Sum of per-monomer ML energies over the system's monomers."""

    name = "ml_intra"

    def __init__(
        self,
        model: Any = None,
        params: Any = None,
        monomer_indices: Sequence[Sequence[int]] | None = None,
        charges: Any = None,
        spins: Any = None,
    ):
        self.model = model
        self.params = params
        self.monomer_indices = monomer_indices
        self.charges = charges
        self.spins = spins

    def neighbor_request(self, system: MolecularSystem):
        return None  # intramolecular, no pair list

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        from mmml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn

        model, params = resolve_ml_model(self, ctx)
        displacement_fn = resolve_displacement_fn(system, ctx)
        jax_z = jnp.asarray(system.Z, dtype=jnp.int32)

        raw_monomers = self.monomer_indices if self.monomer_indices is not None else system.monomer_indices
        if not raw_monomers:
            raise ValueError("ml_intra requires monomer_indices (system.monomer_indices is empty).")
        monomers = [jnp.asarray(m, dtype=jnp.int32) for m in raw_monomers]

        monomer_energy_fn = make_monomer_energy_fn(
            model, params, jax_z, monomers, displacement_fn,
            charges=self.charges, spins=self.spins,
        )

        def energy_fn(R, **kwargs) -> Any:
            return monomer_energy_fn(R)

        return TermFns(jax_energy_fn=energy_fn, ase_contribution=None, neighbor_request=None)
