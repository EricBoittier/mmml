"""Constrained relaxation of IC-scan geometries with ASE ``FixInternals``."""

from __future__ import annotations

import os
from collections.abc import Callable

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.optimize.fire import FIRE

from .config import DegreeOfFreedom, IcScanConfig
from .geometry import measure_dof
from .topology import angles_match, circular_delta_deg

CalculatorFactory = Callable[[], Calculator]

# Match rigid-apply verification: 1° / 0.001 Å after constrained min.
RELAX_ATOL_DEG = 1.0
RELAX_ATOL_BOND_A = 1.0e-3


def fix_internals_constraint(
    dofs: tuple[DegreeOfFreedom, ...],
    coordinates: dict[str, float],
    *,
    names: tuple[str, ...],
) -> FixInternals:
    """Build ``FixInternals`` for the named DoFs at the requested values.

    Only *active* scan axes should be constrained; inactive internals stay free
    so a 1D methyl rotor can relax the amide and the other C–H geometry.
    """

    if not names:
        raise ValueError("constrained relaxation needs at least one active DoF")
    dof_map = {dof.name: dof for dof in dofs}
    bonds: list[list[object]] = []
    angles_deg: list[list[object]] = []
    dihedrals_deg: list[list[object]] = []
    for name in names:
        dof = dof_map[name]
        value = float(coordinates[name])
        indices = list(dof.atoms)
        if dof.kind == "bond":
            bonds.append([value, indices])
        elif dof.kind == "angle":
            angles_deg.append([value, indices])
        else:
            dihedrals_deg.append([value, indices])
    return FixInternals(
        bonds=bonds or None,
        angles_deg=angles_deg or None,
        dihedrals_deg=dihedrals_deg or None,
    )


def _assert_active_dofs_held(
    atoms: Atoms,
    dofs: tuple[DegreeOfFreedom, ...],
    coordinates: dict[str, float],
    *,
    names: tuple[str, ...],
) -> None:
    dof_map = {dof.name: dof for dof in dofs}
    problems: list[str] = []
    for name in names:
        dof = dof_map[name]
        target = float(coordinates[name])
        actual = measure_dof(atoms, dof)
        if dof.kind == "bond":
            if abs(actual - target) > RELAX_ATOL_BOND_A:
                problems.append(
                    f"{name}: requested {target:.6g} Å, got {actual:.6g} Å"
                )
            continue
        if not angles_match(actual, target, atol_deg=RELAX_ATOL_DEG):
            problems.append(
                f"{name}: requested {target:.4g}°, got {actual:.4g}° "
                f"(Δ={circular_delta_deg(actual, target):+.4g}°)"
            )
    if problems:
        raise ValueError(
            "constrained relaxation drifted off the scanned internals:\n  - "
            + "\n  - ".join(problems)
        )


def constrained_relax_atoms(
    atoms: Atoms,
    factory: CalculatorFactory,
    config: IcScanConfig,
    *,
    active_dofs: tuple[str, ...],
    coordinates: dict[str, float],
) -> Atoms:
    """Minimize ``atoms`` with active scan DoFs held by ``FixInternals``."""

    work = atoms.copy()
    work.set_constraint()
    work.calc = factory()
    work.set_constraint(
        fix_internals_constraint(
            config.dofs, coordinates, names=active_dofs
        )
    )
    if config.relax_optimizer == "fire":
        opt = FIRE(
            work,
            logfile=os.devnull,
            maxstep=config.relax_maxstep_A,
        )
    elif config.relax_optimizer == "bfgs":
        opt = BFGS(
            work,
            logfile=os.devnull,
            maxstep=config.relax_maxstep_A,
        )
    else:
        raise ValueError(
            f"unsupported relax_optimizer: {config.relax_optimizer!r}"
        )
    converged = bool(
        opt.run(fmax=config.relax_fmax_ev_A, steps=config.relax_steps)
    )
    work.set_constraint()
    work.info["relax_steps"] = int(getattr(opt, "nsteps", 0))
    work.info["relax_converged"] = converged
    if not converged:
        raise RuntimeError(
            "constrained relaxation did not converge in "
            f"{config.relax_steps} {config.relax_optimizer.upper()} steps "
            f"(fmax={config.relax_fmax_ev_A} eV/Å)"
        )
    _assert_active_dofs_held(
        work, config.dofs, coordinates, names=active_dofs
    )
    return work
