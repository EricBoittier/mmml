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

.. deprecated::
    This module is deprecated. Use :mod:`pycharmm.restraints` instead:

    - ``cons_fix.setup()`` → ``restraints.atoms.fix()``
    - ``cons_fix.turn_off()`` → ``restraints.atoms.fix_turn_off()``

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

import warnings

from pycharmm import restraints


def turn_off(comparison=False):
    """Turn off and clear settings for fix constraints

    .. deprecated::
        Use ``restraints.atoms.fix_turn_off()`` instead.

    Parameters
    ----------
    comparison : bool
         if true, turn off fix contraints on the comparison set

    Returns
    -------
    bool
                True <==> success


    """
    warnings.warn(
        "cons_fix.turn_off() is deprecated, use restraints.atoms.fix_turn_off()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.fix_turn_off(comparison=comparison)


def setup(selection, comparison=False, purge=False,
          bond=False, angle=False, phi=False, imp=False, cmap=False):
    """Configure and turn on fix constraints for the selected atoms

    .. deprecated::
        Use ``restraints.atoms.fix()`` instead.

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
    warnings.warn(
        "cons_fix.setup() is deprecated, use restraints.atoms.fix()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.fix(
        selection=selection,
        comparison=comparison,
        purge=purge,
        bond=bond,
        angle=angle,
        phi=phi,
        imp=imp,
        cmap=cmap
    )
