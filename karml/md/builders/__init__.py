"""System builders: ``SystemSpec`` -> ``MolecularSystem``.

Each builder wraps an existing construction backend (packmol, pyxtal, peptide
builder, template PDB) behind one seam, and is the single place
:class:`~karml.md.system.FFParams` is resolved (decision A). Concrete builders
migrate here from ``karml.interfaces.pycharmmInterface`` and
``karml.cli.run.md_pbc_suite`` in later steps.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from karml.md.builders._topology import (
    molecule_ids_from_bonds,
    monomer_indices_from_mol_id,
)
from karml.md.builders.psf import PsfSystemBuilder
from karml.md.builders.placement import (
    PackmolSystemBuilder,
    PeptideWaterSystemBuilder,
    PyxtalSystemBuilder,
)
from karml.md.system import MolecularSystem, SystemSpec

__all__ = [
    "SystemBuilder",
    "PsfSystemBuilder",
    "PackmolSystemBuilder",
    "PyxtalSystemBuilder",
    "PeptideWaterSystemBuilder",
    "molecule_ids_from_bonds",
    "monomer_indices_from_mol_id",
]


@runtime_checkable
class SystemBuilder(Protocol):
    """Builds an immutable :class:`MolecularSystem` from a :class:`SystemSpec`."""

    name: str

    def build(self, spec: SystemSpec) -> MolecularSystem:
        ...
