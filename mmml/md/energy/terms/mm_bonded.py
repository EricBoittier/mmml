"""CGenFF bonded term for MM atoms (mechanical embedding).

When ``ml_intra`` is restricted to a solute complex via ``ml_resnames``, solvent
molecules otherwise have **no** intramolecular forces (MM nonbonded is
intermolecular-only). This term supplies CHARMM/CGenFF bond/angle/torsion/
improper/Urey–Bradley on atoms outside the ML region so TIP3 (etc.) stay intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax, resolve_displacement_fn
from mmml.md.system import MolecularSystem

__all__ = ["MMBondedTerm"]


@register_term("mm_bonded")
class MMBondedTerm:
    """CGenFF bonded energy on MM atoms (ML-region bonded interactions dropped)."""

    name = "mm_bonded"

    def __init__(
        self,
        ml_atom_indices: Sequence[int] | None = None,
        extra_prm_files: Sequence[str | Path] = (),
        prm_file: str | Path | None = None,
        *,
        topology: Any = None,
        bonded: Any = None,
        urey_k: Any = None,
        urey_r0: Any = None,
    ):
        self.ml_atom_indices = (
            None
            if ml_atom_indices is None
            else np.asarray(list(ml_atom_indices), dtype=np.int32)
        )
        self.extra_prm_files = tuple(Path(p) for p in extra_prm_files)
        self.prm_file = None if prm_file is None else Path(prm_file)
        # Optional prebuilt topology (unit tests / callers that already filtered).
        self.topology = topology
        self.bonded = bonded
        self.urey_k = urey_k
        self.urey_r0 = urey_r0

    def neighbor_request(self, system: MolecularSystem):
        return None

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
            KCAL_MOL_TO_EV,
            bonded_energy_components,
        )
        from mmml.interfaces.pycharmmInterface.mixed_ml_mm import prepare_mm_bonded_system

        displacement_fn = resolve_displacement_fn(system, ctx)

        if self.topology is not None and self.bonded is not None:
            topology = self.topology
            bonded = self.bonded
            urey_k = self.urey_k
            urey_r0 = self.urey_r0
        else:
            if system.psf_path is None:
                raise ValueError(
                    "mm_bonded requires system.psf_path (or an explicit topology)"
                )
            from mmml.interfaces.pycharmmInterface.cgenff_topology import (
                load_cgenff_bonded_from_psf,
            )

            full = load_cgenff_bonded_from_psf(
                system.psf_path,
                system.R,
                prm_file=self.prm_file,
                extra_prm_files=self.extra_prm_files,
                molecule_id=system.mol_id,
            )
            ml_idx = self.ml_atom_indices
            if ml_idx is None:
                ml_idx = np.asarray(
                    ctx.options.get("ml_atom_indices", []), dtype=np.int32
                )
            if ml_idx.size:
                filtered, _mask = prepare_mm_bonded_system(full, ml_idx)
            else:
                filtered = full
            topology = filtered.topology
            bonded = filtered.bonded
            urey_k = filtered.urey_k
            urey_r0 = filtered.urey_r0

        def energy_fn(R, **kwargs) -> Any:
            components = bonded_energy_components(
                jnp.asarray(R),
                topology,
                bonded,
                displacement_fn,
                urey_k=urey_k,
                urey_r0=urey_r0,
            )
            return components["total"] * KCAL_MOL_TO_EV

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )
