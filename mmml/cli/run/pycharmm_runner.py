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

# Nbonds CHARMM script: non-bonding parameters and SHAKE constraints
NBONDS_SCRIPT = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################
! Non-bonding parameters
nbonds atom cutnb 10.0  ctofnb 9.0 ctonnb 8.0 -
fswitch vswitch NBXMOD 5 -
inbfrq -1 imgfrq -1
shake bonh para sele all end
"""


def run_pycharmm_nbonds_minimize(args: Any) -> None:
    """Run PyCHARMM nbonds setup and ABNR minimization."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block

    reset_block()
    pycharmm.lingo.charmm_script(NBONDS_SCRIPT)
    safe_energy_show()
    print("Running PyCHARMM minimize")
    pycharmm_soft()
    pycharmm.minimize.run_abnr(
        nstep=getattr(args, "pycharmm_minimize_steps", 1000),
        tolenr=1e-2,
        tolgrd=1e-2,
    )
    pycharmm_quiet()
    pycharmm.lingo.charmm_script("ENER")
    safe_energy_show()


def run_pycharmm_setup_and_minimize(atoms: Any, args: Any) -> Any:
    """Run PyCHARMM nbonds+minimize if enabled, sync atoms from coor."""
    _do_pycharmm_min = getattr(args, "pycharmm_minimize", None)
    if _do_pycharmm_min is None and getattr(args, "no_pycharmm_minimize", False):
        _do_pycharmm_min = False
    if _do_pycharmm_min is None:
        _do_pycharmm_min = True
    if _do_pycharmm_min:
        run_pycharmm_nbonds_minimize(args)
    else:
        print("Skipping PyCHARMM nbonds/minimize (--pycharmm-minimize False); using PDB positions.")
    atoms.set_positions(coor.get_positions())
    return atoms


def _run_charmm_phase(script: str, atoms: Any, args: Any) -> Any:
    """Common pattern: run CHARMM script, minimize, sync positions."""
    pycharmm.lingo.charmm_script(script)
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


def run_heat(atoms: Any, args: Any) -> Any:
    """Run CHARMM heat phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import heat

    return _run_charmm_phase(heat, atoms, args)


def run_equilibration(atoms: Any, args: Any) -> Any:
    """Run CHARMM equilibration phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import equi

    return _run_charmm_phase(equi, atoms, args)


def run_production(atoms: Any, args: Any) -> Any:
    """Run CHARMM production phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import production

    return _run_charmm_phase(production, atoms, args)
