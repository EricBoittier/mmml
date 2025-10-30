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

"""Get data from CHARMM associated with 'psf' files

Functions
=========
- `get_system_serial` -- get serial version of current charmm/openmm system

Examples
========
>>> import pycharmm
>>> import pycharmm.omm as omm

Get serlialized system
>>> my_sys_xml = omm.get_system_serial()
"""

import ctypes

import pycharmm
import pycharmm.lib as lib


def get_system_serial():
    """Get the serialized version of current charmm/openmm system

       This returns a string of characters in a specific xml format.
       The string can then be read into a python OpenMM simmulation.

    Returns
    -------
    str
        A serialized representation of the current charmm/openmm system

    """
    max_size = lib.charmm.omm_get_system_serial_size()
    buffer = ctypes.create_string_buffer(max_size)
    lib.charmm.omm_get_system_serial(buffer, max_size)
    return buffer.value.decode(errors='ignore')
