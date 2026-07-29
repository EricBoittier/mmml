"""Built-in energy terms.

Importing this package registers the built-in terms in the term registry
(:func:`mmml.md.energy.registry.register_term`). It is imported lazily — pulling
in ``jax`` — so the ``mmml.md`` protocol/dataclass seams stay dependency-light
(see ``docs/md-cg-unification-design.md``).

All energy terms are now extracted: bias/restraint (`smd`, `dihedral`,
`rxncoor`),
`vdw_core`, `mm_bonded`, `mm_nonbonded`, the ML terms (`ml_intra`,
`ml_pep_water`), and the rigid-QCML intermolecular set (`zbl`, `mbd`,
`multipole`).
"""

from __future__ import annotations

from mmml.md.energy.terms.dihedral import DihedralRestraint, DihedralRestraintTerm
from mmml.md.energy.terms.mbd import MBDDispersionTerm
from mmml.md.energy.terms.ml_intra import MLIntramolecularTerm
from mmml.md.energy.terms.ml_mm_elec import MLMMElectrostaticTerm
from mmml.md.energy.terms.ml_pep_water import MLCoreGroupTerm
from mmml.md.energy.terms.mm_bonded import MMBondedTerm
from mmml.md.energy.terms.mm_nonbonded import MMNonbondedTerm
from mmml.md.energy.terms.multipole import MultipoleTerm
from mmml.md.energy.terms.rxncoor import ReactionCoordinateBiasTerm
from mmml.md.energy.terms.smd import SMDBiasTerm
from mmml.md.energy.terms.vdw_core import RepulsiveCoreVdwTerm
from mmml.md.energy.terms.zbl import ZBLTerm

__all__ = [
    "DihedralRestraint",
    "DihedralRestraintTerm",
    "MBDDispersionTerm",
    "MLIntramolecularTerm",
    "MLMMElectrostaticTerm",
    "MLCoreGroupTerm",
    "MMBondedTerm",
    "MMNonbondedTerm",
    "MultipoleTerm",
    "ReactionCoordinateBiasTerm",
    "SMDBiasTerm",
    "RepulsiveCoreVdwTerm",
    "ZBLTerm",
]
