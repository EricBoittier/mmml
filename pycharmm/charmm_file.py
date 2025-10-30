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

"""A class to manipulate files at the fortran level

Classes
=======
- `CharmmFile` -- open and close files with access to unit number and file name
"""

import ctypes

import pycharmm.lib as lib


class CharmmFile:
    """
    A class to manipulate files at the fortran level
    """
    def __init__(self, file_name, file_unit=-1,
                 read_only=True, append=False, formatted=False):
        """class constructor

        :param string file_name: name of the file
        :param int file_unit: associate this unit number with the file, get next unused unit number if -1
        :param bool read_only: open the file in read only mode, no writing allowed
        :param bool append: If the file is written to, should the new content be appended to the end?
        :param bool formatted: Is this file formatted in the fortran sense?
        """
        self.file_name = file_name
        self.file_unit = file_unit
        self.read_only = read_only
        self.append = append
        self.formatted = formatted
        self.is_open = False
        self.open()

    def __del__(self):
        """class destructor

        if the file is open, close it
        """
        if self.is_open:
            self.close()

    def open(self):
        """Open the file

        Returns
        -------
        bool
            True if file is open
        """
        if self.is_open:
            return True

        fn = ctypes.c_char_p(self.file_name.encode())
        len_fn = ctypes.c_int(len(self.file_name))

        to_read = ctypes.c_int(1)
        to_write = ctypes.c_int(0)
        if not self.read_only:
            to_read = ctypes.c_int(0)
            to_write = ctypes.c_int(1)

        to_append = ctypes.c_int(0)
        if self.append:
            to_append = ctypes.c_int(1)

        formatted = ctypes.c_int(0)
        if self.formatted:
            formatted = ctypes.c_int(1)

        new_unit = ctypes.c_int(self.file_unit)

        # Open the file by calling a charmm fortran routine
        charmm_unit = lib.charmm.charmm_file_open(
            fn, len_fn,
            to_read, to_write, to_append,
            formatted,
            new_unit)

        if not charmm_unit == -1:
            self.file_unit = charmm_unit
            self.is_open = True

        # consider throwing an error if charmm_unit == -1

        return self.is_open

    def close(self):
        """Close the file

        Returns
        -------
        bool
            True if file is closed
        """
        success_bit = 0
        if self.is_open:
            unit = ctypes.c_int(self.file_unit)
            success_bit = lib.charmm.charmm_file_close(unit)

        if success_bit == 1:
            self.is_open = False

        # otherwise, think about throwing a specific error here
        return not self.is_open
