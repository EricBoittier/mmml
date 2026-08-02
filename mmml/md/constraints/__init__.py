"""Holonomic constraints (SHAKE/RATTLE), separate from restraints.

A restraint adds an energy penalty and lets the coordinate move; a constraint
removes the degree of freedom outright. See ``mmml.md.restraints`` for the
former.
"""

from mmml.md.constraints.rattle import (
    MolecularConstraints,
    constraint_residuals,
    rattle_velocities,
    shake_positions,
    tip3_rigid_constraints,
)

__all__ = [
    "MolecularConstraints",
    "constraint_residuals",
    "rattle_velocities",
    "shake_positions",
    "tip3_rigid_constraints",
]
