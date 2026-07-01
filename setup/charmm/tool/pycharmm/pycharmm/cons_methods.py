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

"""Functions to configure various constraints/restraints covering:

- cons dihe

- cons cldh

- cons ic

- cons droplet

- cons hmcm

Corresponds to CHARMM command `CONS`
  
See CHARMM documentation [CONS](<https://academiccharmm.org/documentation/version/c47b1/cons>)
for more information

Functions
=========
- `dihe` -- set up dihedral restraints 
- `ic` -- set up internal coordinates (IC) restraints 
- `droplet` -- set up the quartic droplet potential

Examples
========
Apply a dihedral potential on four selected atoms
>>> import pycharmm.cons_methods as cons_methods
>>> cons_methods.dihe(selection='bynum 7 9 15 17',minimum=-60,force=1.0, width=0)
 

"""
import ctypes
import pycharmm.lib as lib
import pycharmm.script

def dihe(selection='', cldh=False, force=0, **kwargs):
    """Set-up/turn-off dihedral angle restraints

    See [CONS DIHE](<https://academiccharmm.org/documentation/version/c47b1/cons#Dihedral>)
    for more information

    Parameters
    ----------
    selection : string
         ['bynum int int int int'] ['4x(segid resid iupac)'] ['4x(resnumber iupac)']
    force : real
         force constant
    **kwargs : dict
        possible keyword arguments are:

        minimum : real

        period : int

        width : real 

        comp : bool  

        main : bool
    """

    if not cldh and len(selection)>0:
        cons_command = 'cons dihe ' + str(selection)
        cons_dihe = pycharmm.script.CommandScript(cons_command,
                                                  force=force,
                                                  **kwargs)
    else:
        cons_dihe =  pycharmm.script.CommandScript('cons cldh')
    cons_dihe.run()

def ic(**kwargs):
    """Impose internal coordinate restraints

    See [CONS IC](<https://academiccharmm.org/documentation/version/c47b1/cons#InternalCoord>)
    for more information

    Parameters
    ----------
    **kwargs: dict
        possible keyword arguments are:

        bond: real  

        angle: real

        dihedral : real

        improper : real

        exponent : int 

        upper : bool

    """
    cons_ic = pycharmm.script.CommandScript('cons ic', **kwargs)
    cons_ic.run()

def droplet(**kwargs):
    """Impose quartic droplet restraints

    See [CONS DROPlet](<https://academiccharmm.org/documentation/version/c47b1/cons#QuarticDroplet>)
    for more information

    Parameters
    ----------
    **kwargs : dict
        possible keyword arguments are:

        force : real

        exponent : int

        nomass : bool

    """
    cons_droplet = pycharmm.script.CommandScript('cons droplet', **kwargs)
    cons_droplet.run()

