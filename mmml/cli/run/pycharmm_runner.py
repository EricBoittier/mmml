"""
PyCHARMM heat, equilibration, and production routines.

Extracted from run_sim.py to separate PyCHARMM-specific code paths.
"""
from __future__ import annotations

from typing import Any

import pycharmm

from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    coor,
    pycharmm_quiet,
    pycharmm_soft,
    safe_energy_show,
)


def run_heat(atoms: Any, args: Any) -> Any:
    """Run CHARMM heat phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import heat

    pycharmm.lingo.charmm_script(heat)
    atoms.set_positions(coor.get_positions())
    safe_energy_show()
    pycharmm_soft()
    pycharmm.minimize.run_abnr(
        nstep=getattr(args, "pycharmm_minimize_steps", 1000),
        tolenr=1e-2,
        tolgrd=1e-2,
    )
    pycharmm_quiet()
    safe_energy_show()
    pycharmm.lingo.charmm_script("ENER")
    atoms.set_positions(coor.get_positions())
    return atoms


def run_equilibration(atoms: Any, args: Any) -> Any:
    """Run CHARMM equilibration phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import equi

    pycharmm.lingo.charmm_script(equi)
    atoms.set_positions(coor.get_positions())
    safe_energy_show()
    pycharmm_soft()
    pycharmm.minimize.run_abnr(
        nstep=getattr(args, "pycharmm_minimize_steps", 1000),
        tolenr=1e-2,
        tolgrd=1e-2,
    )
    pycharmm_quiet()
    safe_energy_show()
    pycharmm.lingo.charmm_script("ENER")
    atoms.set_positions(coor.get_positions())
    return atoms


def run_production(atoms: Any, args: Any) -> Any:
    """Run CHARMM production phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import production

    pycharmm.lingo.charmm_script(production)
    atoms.set_positions(coor.get_positions())
    safe_energy_show()
    pycharmm_soft()
    pycharmm.minimize.run_abnr(
        nstep=getattr(args, "pycharmm_minimize_steps", 1000),
        tolenr=1e-2,
        tolgrd=1e-2,
    )
    pycharmm_quiet()
    safe_energy_show()
    pycharmm.lingo.charmm_script("ENER")
    atoms.set_positions(coor.get_positions())
    return atoms
