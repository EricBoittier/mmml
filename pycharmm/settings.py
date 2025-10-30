# pyCHARMM: molecular dynamics in python with CHARMM
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

"""Functions to change CHARMM print level (`PRNLev`), warning level (`WRNLev`) 
and bomb level (`BOMBlev`)

See CHARMM documentation [miscom](<https://academiccharmm.org/documentation/version/c47b1/miscom>)
for more information

Functions
=========

- `set_verbosity` -- set the CHARMM library's verbosity level
- `set_warn_level` -- set the CHARMM library's warning level
- `set_bomb_level` -- set the CHARMM library's bomb level

Examples
========
>>> import pycharmm
>>> import pycharmm.settings as settings

The following command is equivalent to CHARMM command `PRNLev 0`
>>> settings.set_verbosity(0)
"""

import ctypes
import pycharmm.lib as lib


# set charmm's verbosity
def set_verbosity(level):
    """change verbosity of CHARMM library

    Parameters
    ----------
    level : int
            the new verbosity level desired

    Returns
    -------
    old_level : int
                old verbosity level
    """
    level = ctypes.c_int(level)
    old_level = lib.charmm.stream_set_prnlev(ctypes.byref(level))
    return old_level


# set charmm's warning level
def set_warn_level(level):
    """change CHARMM's warning level

    Parameters
    ----------
    level : int
            the new warning level desired

    Returns
    -------
    old_level : int
                old warning level
    """
    level = ctypes.c_int(level)
    old_level = lib.charmm.stream_set_wrnlev(ctypes.byref(level))
    return old_level


# set charmm's bomb level
def set_bomb_level(level):
    """change CHARMM's bomb level

    Parameters
    ----------
    level : int
            the new bomb level desired

    Returns
    -------
    old_level : int
                old bomb level
    """
    level = ctypes.c_int(level)
    old_level = lib.charmm.stream_set_bomlev(ctypes.byref(level))
    return old_level
