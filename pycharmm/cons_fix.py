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

"""Functions to configure fix constraints

Corresponds to CHARMM command `CONS FIX`

See CHARMM documentation [CONS FIX](<https://academiccharmm.org/documentation/version/c47b1/cons#FixedAtom>)
for more information

Functions
=========
- `turn_off` -- turn off fix constraints
- `setup` -- turn on fix constraints

Examples
=========
Fix all atoms in a segment named PROT
>>> import pycharmm.cons_fix as cons_fix
>>> cons_fix.setup(pycharmm.SelectAtoms(seg_id='PROT'))


"""

import ctypes

import pycharmm
import pycharmm.lib as lib


def turn_off(comparison=False):
    """Turn off and clear settings for fix constraints

    Parameters
    ----------
    comparison : bool 
         if true, turn off fix contraints on the comparison set

    Returns
    -------
    bool
                True <==> success


    """
    none_selected = pycharmm.SelectAtoms()
    status = setup(none_selected, comparison)
    return status


def setup(selection, comparison=False, purge=False,
          bond=False, angle=False, phi=False, imp=False, cmap=False):
    """Configure and turn on fix constraints for the selected atoms

    Parameters
    ----------
    selection : pycharmm.SelectAtoms  
                selection[i] == 1 <=> apply constraints to atom i
    comparison : bool
                if true, do constraints on comparison set instead of main set
    purge : bool 
                if true, use the purge option which modified the PSF irrevocably
    bond : bool 
                if true, use the bond option
    angle: bool 
                if true, use the angle option
    phi: bool 
                if true, use the phi option
    imp: bool 
                if true, use the imp option
    cmap: bool 
                if true, use the cmap option

    Returns
    -------
    bool
                True <==> success

    """
    c_sel = selection.as_ctypes()

    c_comp = ctypes.c_int(comparison)
    c_purge = ctypes.c_int(purge)
    c_bond = ctypes.c_int(bond)
    c_angle = ctypes.c_int(angle)
    c_phi = ctypes.c_int(phi)
    c_imp = ctypes.c_int(imp)
    c_cmap = ctypes.c_int(cmap)

    status = lib.charmm.cons_fix_setup(c_sel,
                                       ctypes.byref(c_comp),
                                       ctypes.byref(c_purge),
                                       ctypes.byref(c_bond),
                                       ctypes.byref(c_angle),
                                       ctypes.byref(c_phi),
                                       ctypes.byref(c_imp),
                                       ctypes.byref(c_cmap))
    status = bool(status)
    return status
