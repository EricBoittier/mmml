"""Holonomic constraints (SHAKE/RATTLE), separate from restraints.

A restraint adds an energy penalty and lets the coordinate move; a constraint
removes the degree of freedom outright. See ``mmml.md.restraints`` for the
former.
"""

from mmml.md.constraints.rattle import (
    MolecularConstraints,
    constrained_nve,
    constrained_velocity_verlet,
    constraint_residuals,
    maybe_wrap_rigid_water,
    molecular_virial_decomposition,
    rattle_velocities,
    rigid_water_spec_from_args,
    shake_positions,
    tip3_rigid_constraints,
    wrap_apply_fn_with_constraints,
)

__all__ = [
    "MolecularConstraints",
    "constrained_nve",
    "constrained_velocity_verlet",
    "constraint_residuals",
    "maybe_wrap_rigid_water",
    "molecular_virial_decomposition",
    "rattle_velocities",
    "rigid_water_spec_from_args",
    "shake_positions",
    "tip3_rigid_constraints",
    "wrap_apply_fn_with_constraints",
]
