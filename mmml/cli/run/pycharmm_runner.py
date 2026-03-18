"""
PyCHARMM heat, equilibration, and production routines.

Extracted from run_sim.py to separate PyCHARMM-specific code paths.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import pycharmm
from rich.console import Console
from rich.panel import Panel

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
    pycharmm_soft()
    pycharmm.lingo.charmm_script(NBONDS_SCRIPT)
    safe_energy_show()
    Console().print(Panel(
        f"ABNR minimization ({getattr(args, 'pycharmm_minimize_steps', 1000)} steps)",
        title="[bold cyan]PyCHARMM[/bold cyan]",
        border_style="cyan",
    ))

    pycharmm.minimize.run_abnr(
        nstep=getattr(args, "pycharmm_minimize_steps", 1000),
        tolenr=1e-2,
        tolgrd=1e-2,
    )
    
    pycharmm.lingo.charmm_script("ENER")
    safe_energy_show()
    pycharmm_quiet()


def run_pycharmm_setup_and_minimize(
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
) -> Any:
    """Run PyCHARMM nbonds+minimize if enabled, sync atoms from coor."""
    _do_pycharmm_min = getattr(args, "pycharmm_minimize", None)
    if _do_pycharmm_min is None and getattr(args, "no_pycharmm_minimize", False):
        _do_pycharmm_min = False
    if _do_pycharmm_min is None:
        _do_pycharmm_min = True
    if _do_pycharmm_min:
        run_pycharmm_nbonds_minimize(args)
    else:
        Console().print(Panel(
            "Skipping nbonds/minimize (--no-pycharmm-minimize); using PDB positions.",
            title="[bold yellow]PyCHARMM[/bold yellow]",
            border_style="yellow",
        ))
    atoms.set_positions(coor.get_positions())
    if show_frame is not None:
        show_frame(atoms, 0, "pycharmm_min")
    return atoms


def _run_charmm_phase(
    script: str,
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
    phase_step: int = 0,
) -> Any:
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
    if show_frame is not None:
        show_frame(atoms, phase_step, "pycharmm")
    return atoms



def run_heat(
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
) -> Any:
    """Run CHARMM heat phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import heat

    Console().print(Panel("Running heat phase", title="[bold cyan]CHARMM Heat[/bold cyan]", border_style="cyan"))
    return _run_charmm_phase(heat, atoms, args, show_frame=show_frame, phase_step=1)


def run_equilibration(
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
) -> Any:
    """Run CHARMM equilibration phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import equi

    Console().print(Panel("Running equilibration", title="[bold cyan]CHARMM Equilibration[/bold cyan]", border_style="cyan"))
    return _run_charmm_phase(equi, atoms, args, show_frame=show_frame, phase_step=2)


def run_production(
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
) -> Any:
    """Run CHARMM production phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import production

    Console().print(Panel("Running production", title="[bold cyan]CHARMM Production[/bold cyan]", border_style="cyan"))
    return _run_charmm_phase(production, atoms, args, show_frame=show_frame, phase_step=3)

def run_dyna(
    atoms: Any,
    args: Any,
    show_frame: Optional[Callable[[Any, int, str], None]] = None,
    ndcd: int = 1,
) -> Any:
    """Run CHARMM production phase."""
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import dyna
    import numpy as np
    iseed = np.random.randint(1, 1000000)
    dyna = dyna.format(iseed=iseed, NDCD=ndcd)
    Console().print(Panel("Running dyna", title="[bold cyan]CHARMM dyna[/bold cyan]", border_style="cyan"))
    return _run_charmm_phase(dyna, atoms, args, show_frame=show_frame, phase_step=3)



"""
PSF helpers
- `get_natom` -- get the current total number of atoms from CHARMM
- `get_nres` -- get the current total number of residues from CHARMM
- `get_nseg` -- get the current total number of segments from CHARMM
- `get_ngrp` -- get the current total number of groups from CHARMM
- `get_iac` -- export a copy of iac (param type codes)
- `get_amass` -- export a copy of amass (atom masses)
- `get_charges` -- export a copy of cg (atom charges)
- `get_ibase` -- export a copy of ibase (last atom of each residue)
- `get_atype` -- export a copy of atype (atom name array)
- `get_res` -- export a copy of res (residue name array)
- `get_resid` -- export a copy of resid (residue identifier array)
- `get_segid` -- export a copy of segid (segment identifier array)
- `get_nictot` -- export a copy of nictot (nres for each seg)
- `get_igpbs` -- export a copy of igpbs (pointer for 1st atom in each group)
- `get_igptyp` -- export a copy of gptyp (code type of each group)
"""
import numpy as np

def get_charge_array() -> np.ndarray:
    """Get the charge array from the PSF file."""
    return np.array(pycharmm.psf.get_charges())

def get_atomtype_array() -> np.ndarray:
    """Get the atomtype array from the PSF file."""
    return np.array(pycharmm.psf.get_atype())

def get_resid_array() -> np.ndarray:
    """Get the resid array from the PSF file."""
    return np.array(pycharmm.psf.get_resid())

def get_segid_array() -> np.ndarray:
    """Get the segid array from the PSF file."""
    return np.array(pycharmm.psf.get_segid())

def get_nictot_array() -> np.ndarray:
    """Get the nictot array from the PSF file."""
    return np.array(pycharmm.psf.get_nictot())

def get_igpbs_array() -> np.ndarray:
    """Get the igpbs array from the PSF file."""
    return np.array(pycharmm.psf.get_igpbs())

def get_igptyp_array() -> np.ndarray:
    """Get the igptyp array from the PSF file."""
    return np.array(pycharmm.psf.get_igptyp())