"""Built-in energy terms.

Importing this package registers the built-in terms in the term registry
(:func:`karml.md.energy.registry.register_term`). It is imported lazily — pulling
in ``jax`` — so the ``karml.md`` protocol/dataclass seams stay dependency-light
(see ``docs/md-cg-unification-design.md``).

All energy terms are now extracted: bias/restraint (`smd`, `dihedral`),
`vdw_core`, `mm_nonbonded`, `mm_bonded` (CGenFF bonded for MM region), the ML
terms (`ml_intra`, `ml_pep_water`), and the rigid-QCML intermolecular set
(`zbl`, `mbd`, `multipole`).
"""

from __future__ import annotations

from karml.md.energy.terms.dihedral import DihedralRestraint, DihedralRestraintTerm
from karml.md.energy.terms.mbd import MBDDispersionTerm
from karml.md.energy.terms.ml_intra import MLIntramolecularTerm
from karml.md.energy.terms.ml_mm_elec import MLMMElectrostaticTerm
from karml.md.energy.terms.ml_mm_pol import MLMMPolarisationTerm
from karml.md.energy.terms.ml_pep_water import MLCoreGroupTerm
from karml.md.energy.terms.mm_bonded import MMBondedTerm
from karml.md.energy.terms.mm_nonbonded import MMNonbondedTerm
from karml.md.energy.terms.multipole import MultipoleTerm
from karml.md.energy.terms.rxncoor import ReactionCoordinateBiasTerm
from karml.md.energy.terms.smd import SMDBiasTerm
from karml.md.energy.terms.vdw_core import RepulsiveCoreVdwTerm
from karml.md.energy.terms.zbl import ZBLTerm

__all__ = [
    "DihedralRestraint",
    "DihedralRestraintTerm",
    "MBDDispersionTerm",
    "MLIntramolecularTerm",
    "MLMMElectrostaticTerm",
    "MLMMPolarisationTerm",
    "MLCoreGroupTerm",
    "MMBondedTerm",
    "MMNonbondedTerm",
    "MultipoleTerm",
    "ReactionCoordinateBiasTerm",
    "SMDBiasTerm",
    "RepulsiveCoreVdwTerm",
    "ZBLTerm",
]
