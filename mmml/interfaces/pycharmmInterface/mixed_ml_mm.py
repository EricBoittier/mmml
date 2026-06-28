"""Mixed ML + MM embedding: one ML molecule with CGENFF-MM environment.

Long-term goal
--------------
Establish a clean embedding between ML and MM atoms:

- **ML region** — internal energy/forces from a neural potential (e.g. PhysNet).
- **MM region** — CGENFF bonded terms (this PR) plus nonbonded terms (future).
- **Boundary** — cross terms (ML–MM electrostatics, link-atom schemes) are not
  implemented here; bonded interactions spanning ML/MM are excluded until an
  explicit coupling model is added.

This module wires the CGENFF bonded JAX path from :mod:`cgenff_bonded` into a
single energy callable suitable for ASE/JAX-MD calculators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
from jax import Array

from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    KCAL_MOL_TO_EV,
    bonded_energy_and_forces,
    build_bonded_energy_fn,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    CgenffBondedSystem,
    filter_bonded_topology_for_mm,
    load_cgenff_bonded_from_charmm_files,
    mm_atom_mask_complement,
)


@dataclass(frozen=True, slots=True)
class MixedMlMmConfig:
    """Selection for a single ML molecule embedded in a CGENFF MM environment."""

    ml_atom_indices: tuple[int, ...]
    energy_unit: str = "eV"

    def mm_atom_indices(self, n_atoms: int) -> tuple[int, ...]:
        ml = set(self.ml_atom_indices)
        return tuple(i for i in range(n_atoms) if i not in ml)


@dataclass(frozen=True, slots=True)
class MixedMlMmEnergyBreakdown:
    ml_energy: Array
    mm_bonded_energy: Array
    mm_bonded_components: dict[str, Array]
    total_energy: Array
    ml_forces: Array
    mm_bonded_forces: Array
    total_forces: Array


MlEnergyFn = Callable[[Array], tuple[Array, Array]]


def prepare_mm_bonded_system(
    system: CgenffBondedSystem,
    ml_atom_indices: Sequence[int],
) -> tuple[CgenffBondedSystem, Array]:
    """Filter bonded topology to MM-only interactions for embedding."""
    mm_mask = mm_atom_mask_complement(ml_atom_indices, system.n_atoms)
    topology, bonded = filter_bonded_topology_for_mm(
        system.topology,
        system.bonded,
        mm_mask,
    )
    filtered = CgenffBondedSystem(
        positions=system.positions,
        topology=topology,
        bonded=bonded,
        atom_types=system.atom_types,
        charges=system.charges,
    )
    return filtered, mm_mask


def build_mixed_ml_mm_energy_fn(
    system: CgenffBondedSystem,
    config: MixedMlMmConfig,
    ml_energy_fn: MlEnergyFn,
) -> Callable[[Array], MixedMlMmEnergyBreakdown]:
    """Combine ML internal energy with CGENFF bonded MM terms.

    Parameters
    ----------
    system
        Full-system CGENFF bonded topology (all atoms).
    config
        ML atom indices and output energy unit.
    ml_energy_fn
        ``(positions) -> (energy, forces)`` for the ML region.  Forces must
        have shape ``(n_ml_atoms, 3)``; they are scattered into the full system.
    """
    mm_system, mm_mask = prepare_mm_bonded_system(system, config.ml_atom_indices)
    mm_bonded_fn = build_bonded_energy_fn(
        mm_system.topology,
        mm_system.bonded,
        energy_unit=config.energy_unit,
    )
    ml_indices = jnp.asarray(config.ml_atom_indices, dtype=jnp.int32)

    def evaluate(positions: Array) -> MixedMlMmEnergyBreakdown:
        ml_e, ml_f_local = ml_energy_fn(positions[ml_indices])
        mm_components, mm_f = mm_bonded_fn(positions)
        mm_e = mm_components["total"]

        ml_f = jnp.zeros_like(positions)
        ml_f = ml_f.at[ml_indices].set(ml_f_local)

        total_e = ml_e + mm_e
        total_f = ml_f + mm_f
        return MixedMlMmEnergyBreakdown(
            ml_energy=ml_e,
            mm_bonded_energy=mm_e,
            mm_bonded_components=mm_components,
            total_energy=total_e,
            ml_forces=ml_f,
            mm_bonded_forces=mm_f,
            total_forces=total_f,
        )

    return evaluate


def evaluate_mixed_ml_mm(
    positions: Array,
    system: CgenffBondedSystem,
    config: MixedMlMmConfig,
    ml_energy_fn: MlEnergyFn,
) -> MixedMlMmEnergyBreakdown:
    """One-shot mixed ML/MM energy evaluation."""
    return build_mixed_ml_mm_energy_fn(system, config, ml_energy_fn)(positions)


def load_mixed_system_from_charmm_files(
    pdb_file: str,
    *,
    residue_name: str | None = None,
    ml_residue_index: int = 0,
    atoms_per_residue: int | None = None,
) -> tuple[CgenffBondedSystem, MixedMlMmConfig]:
    """Load a PDB and mark one residue copy as the ML region.

    For a single-residue PDB, ``ml_residue_index=0`` marks all atoms as ML.
    For multi-residue systems, supply ``atoms_per_residue`` (uniform residue size).
    """
    system = load_cgenff_bonded_from_charmm_files(
        pdb_file,
        residue_name=residue_name,
    )
    n_atoms = system.n_atoms
    if atoms_per_residue is None:
        if ml_residue_index != 0:
            raise ValueError(
                "atoms_per_residue is required when ml_residue_index != 0"
            )
        ml_indices = tuple(range(n_atoms))
    else:
        start = ml_residue_index * atoms_per_residue
        stop = start + atoms_per_residue
        if stop > n_atoms:
            raise ValueError(
                f"ML residue slice [{start}:{stop}) exceeds n_atoms={n_atoms}"
            )
        ml_indices = tuple(range(start, stop))

    config = MixedMlMmConfig(
        ml_atom_indices=ml_indices,
        energy_unit="eV",
    )
    return system, config


__all__ = [
    "KCAL_MOL_TO_EV",
    "MixedMlMmConfig",
    "MixedMlMmEnergyBreakdown",
    "MlEnergyFn",
    "build_mixed_ml_mm_energy_fn",
    "evaluate_mixed_ml_mm",
    "load_mixed_system_from_charmm_files",
    "prepare_mm_bonded_system",
]
