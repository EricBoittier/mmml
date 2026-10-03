# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions to configure SHAKE constraints for bond and angle fixing.

SHAKE is a method of fixing bond lengths and, optionally, bond angles
during dynamics, minimization, and coordinate modification. When hydrogens
are present in a structure, using SHAKE on bonds allows a two-fold
increase in the dynamics step size.

Corresponds to CHARMM command `SHAKe`
See [SHAKE documentation](https://academiccharmm.org/documentation/version/c47b1/cons#SHAKE)

Reference: J. Comp. Phys. 23:327 (1977)

Functions
=========
Main Functions
--------------
- `on` -- configure and turn on SHAKE constraints
- `off` -- turn off SHAKE constraints

Convenience Functions
---------------------
- `on_hydrogen_bonds` -- fix all bonds involving hydrogens (BONH)
- `on_all_bonds` -- fix all bonds (BOND)
- `on_fast` -- use fast vector/parallel SHAKE algorithm

Parameters
==========
Common parameters for SHAKE:

- `bonh` : bool - fix all bonds involving hydrogens
- `bond` : bool - fix all bonds
- `angh` : bool - fix all angles involving hydrogens
- `angl` : bool - fix all angles
- `tol` : float - allowed relative deviation (default: 1e-10)
- `maxiter` : int - maximum SHAKE iterations (default: 500)
- `param` : bool - use parameter table instead of current coordinates
- `fast` : bool - use fast vector/parallel algorithm
- `water` : str - water residue name for fast SHAKE

Examples
========
>>> import pycharmm.shake as shake

Fix all bonds involving hydrogens (most common for 2fs timestep)
>>> shake.on_hydrogen_bonds()

Fix all bonds with fast algorithm for water
>>> shake.on_fast(water='TIP3')

Turn off SHAKE
>>> shake.off()

Custom SHAKE settings
>>> shake.on(bonh=True, tol=1e-8, maxiter=1000)
"""

import ctypes

import pycharmm
from pycharmm.loader import lib
import pycharmm.psf as psf


def off():
    """Turn off SHAKE.

    Returns
    -------
    bool
        True if successful.
    """
    status = lib.shake_off()
    return status


def on(islct=None, jslct=None,
       comparison=False, param=False,
       fast=False, water=None,
       tol=1.0e-10, maxiter=500, scale=1.0,
       bonh=False, bond=False, angh=False, angl=False,
       noreset=False):
    """Configure and turn on SHAKE constraints for the selected atoms

    Parameters
    ----------
    islct : list[int]
            islct(i) & jslct(j) .eq. 1 <=> constrain bond/angle btn atoms i & j
    jslct : list[int]
            see islct
    comparison : bool
                 constrain the comparison set instead of the main set
    param : bool
            use bond dists from param table instead of current coords
    fast : bool
           use vector parallel algorithm with diff assumptions
    water : string
            name of the water residue for the shake fast algorithm
    tol : float, default = 1e-10
          allowed relative deviations from the reference values
    maxiter : int, default = 500
              max iterations SHAKE tries before giving up
    scale : float, default = 1.0
            convergence scale factor for SHAKEA
    bonh : bool, default = False
           all bonds involving hydrogens are fixed
    bond : bool, default = False
           all bonds are fixed
    angh : bool, default = False
           all angles involving hydrogen are fixed
    angl : bool, default = False
           all angles are fixed if
    noreset : bool, default = False
              do not reset counters to 0 during setup

    Returns
    -------
    bool
        True if successful.
    """
    if islct is None:
        islct = pycharmm.SelectAtoms().all_atoms()

    c_isel = islct.as_ctypes()

    if jslct is None:
        jslct = pycharmm.SelectAtoms().all_atoms()

    c_jsel = jslct.as_ctypes()

    c_comp = ctypes.c_int(comparison)
    c_param = ctypes.c_int(param)
    c_fast = ctypes.c_int(fast)
    c_bonh = ctypes.c_int(bonh)
    c_bond = ctypes.c_int(bond)
    c_angh = ctypes.c_int(angh)
    c_angl = ctypes.c_int(angl)
    c_noreset = ctypes.c_int(noreset)

    if water is None:
        water_idx = 0
    else:
        water_idx = psf.get_res_idx(water)

    c_water = ctypes.c_int(water_idx)

    c_tol = ctypes.c_double(tol)
    c_maxiter = ctypes.c_int(maxiter)
    c_scale = ctypes.c_double(scale)

    status = lib.shake_on(c_isel, c_jsel,
                                 ctypes.byref(c_comp),
                                 ctypes.byref(c_param),
                                 ctypes.byref(c_fast),
                                 ctypes.byref(c_water),
                                 ctypes.byref(c_tol),
                                 ctypes.byref(c_maxiter),
                                 ctypes.byref(c_scale),
                                 ctypes.byref(c_bonh),
                                 ctypes.byref(c_bond),
                                 ctypes.byref(c_angh),
                                 ctypes.byref(c_angl),
                                 ctypes.byref(c_noreset))
    status = bool(status)
    return status


def on_hydrogen_bonds(tol=1.0e-10, maxiter=500, param=False, **kwargs):
    """Turn on SHAKE for all bonds involving hydrogens (BONH).

    This is the most common SHAKE setting, allowing a 2 fs timestep
    in molecular dynamics simulations.

    Parameters
    ----------
    tol : float, default = 1e-10
        allowed relative deviations from the reference values
    maxiter : int, default = 500
        maximum iterations SHAKE tries before giving up
    param : bool, default = False
        use bond distances from parameter table instead of current coordinates
    **kwargs : optional
        additional keyword arguments passed to `on()`

    Returns
    -------
    bool
        True if successful.

    Examples
    --------
    >>> import pycharmm.shake as shake
    >>> shake.on_hydrogen_bonds()
    """
    return on(bonh=True, tol=tol, maxiter=maxiter, param=param, **kwargs)


def on_all_bonds(tol=1.0e-10, maxiter=500, param=False, **kwargs):
    """Turn on SHAKE for all bonds (BOND).

    Fixes all bond lengths, not just those involving hydrogens.

    Parameters
    ----------
    tol : float, default = 1e-10
        allowed relative deviations from the reference values
    maxiter : int, default = 500
        maximum iterations SHAKE tries before giving up
    param : bool, default = False
        use bond distances from parameter table instead of current coordinates
    **kwargs : optional
        additional keyword arguments passed to `on()`

    Returns
    -------
    bool
        True if successful.

    Examples
    --------
    >>> import pycharmm.shake as shake
    >>> shake.on_all_bonds()
    """
    return on(bond=True, tol=tol, maxiter=maxiter, param=param, **kwargs)


def on_fast(water=None, tol=1.0e-10, maxiter=500, param=False, **kwargs):
    """Turn on fast vector/parallel SHAKE algorithm.

    Uses the optimized SHAKE algorithm for better performance. This assumes
    only bonds between heavy atoms and hydrogens are constrained, except for
    water molecules specified by the water parameter.

    The fast algorithm is about 5x faster than scalar SHAKE for heavy
    atom-hydrogen bonds. For water molecules, a special direct inversion
    method is used which is 25-30% faster than iterative SHAKE.

    Parameters
    ----------
    water : str, optional
        name of the water residue (e.g., 'TIP3', 'SPC') for special
        water SHAKE treatment. Water molecules must have exactly two
        X-H bonds and one H-H bond.
    tol : float, default = 1e-10
        allowed relative deviations from the reference values
    maxiter : int, default = 500
        maximum iterations SHAKE tries before giving up
    param : bool, default = False
        use bond distances from parameter table instead of current coordinates
    **kwargs : optional
        additional keyword arguments passed to `on()`

    Returns
    -------
    bool
        True if successful.

    Examples
    --------
    >>> import pycharmm.shake as shake
    >>> shake.on_fast(water='TIP3')
    """
    return on(fast=True, water=water, bonh=True, tol=tol, maxiter=maxiter,
              param=param, **kwargs)
