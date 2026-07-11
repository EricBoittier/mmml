"""Built-in energy terms.

Importing this package registers the built-in terms in the term registry
(:func:`mmml.md.energy.registry.register_term`). It is imported lazily — pulling
in ``jax`` — so the ``mmml.md`` protocol/dataclass seams stay dependency-light
(see ``docs/md-cg-unification-design.md``).

Extraction status (§9/§11): bias/restraint terms first (no CHARMM/checkpoint
dependency); ``ml_intra`` / ``ml_pep_water`` / ``mm_nonbonded`` / ``vdw_core``
follow.
"""

from __future__ import annotations

from mmml.md.energy.terms.dihedral import DihedralRestraint, DihedralRestraintTerm
from mmml.md.energy.terms.smd import SMDBiasTerm
from mmml.md.energy.terms.vdw_core import RepulsiveCoreVdwTerm

__all__ = [
    "DihedralRestraint",
    "DihedralRestraintTerm",
    "SMDBiasTerm",
    "RepulsiveCoreVdwTerm",
]
