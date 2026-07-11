"""System builders: ``SystemSpec`` -> ``MolecularSystem``.

Each builder wraps an existing construction backend (packmol, pyxtal, peptide
builder, template PDB) behind one seam, and is the single place
:class:`~mmml.md.system.FFParams` is resolved (decision A). Concrete builders
migrate here from ``mmml.interfaces.pycharmmInterface`` and
``mmml.cli.run.md_pbc_suite`` in later steps.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mmml.md.system import MolecularSystem, SystemSpec

__all__ = ["SystemBuilder"]


@runtime_checkable
class SystemBuilder(Protocol):
    """Builds an immutable :class:`MolecularSystem` from a :class:`SystemSpec`."""

    name: str

    def build(self, spec: SystemSpec) -> MolecularSystem:
        ...
