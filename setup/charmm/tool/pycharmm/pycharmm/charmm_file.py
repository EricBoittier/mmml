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
- `CharmmFile` -- open and close files with access to unit number and file name.
  Supports the context manager protocol (``with`` statement) for
  automatic cleanup.
"""

import ctypes

# import pycharmm.loader as lib
from pycharmm.loader import lib


def c_api_path_buffer(file_name: str) -> tuple[ctypes.Array, ctypes.c_int]:
    """Stable path buffer + length for CHARMM ``bind(c)`` file APIs.

    ``ctypes.c_char_p(path.encode())`` is unsafe: the backing bytes can be
    collected before Fortran reads the pointer (segfault in ``read_rtf_file``).
    """
    buf = ctypes.create_string_buffer(file_name.encode())
    return buf, ctypes.c_int(len(file_name))


def c_api_string_buffer(text: str) -> tuple[ctypes.Array, ctypes.c_int]:
    """Stable buffer + length for arbitrary CHARMM ``bind(c)`` string APIs."""
    encoded = text.encode()
    buf = ctypes.create_string_buffer(encoded)
    return buf, ctypes.c_int(len(encoded))


def _resolve_charmm_fortran_path(file_name, *, read_only, append):
    try:
        from karml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path
    except ImportError:
        return file_name, None
    return charmm_fortran_path(
        file_name,
        for_write=not read_only,
        append=append,
    )


class CharmmFile:
    """A class to manipulate files at the Fortran level.

    Can be used as a context manager::

        with CharmmFile('traj.dcd', file_unit=40, read_only=False) as dcd:
            DynamicsScript(..., iuncrd=dcd.file_unit).run()
        # file is automatically closed here

    A closed file can be reopened in a different mode::

        dcd.open(read_only=True)
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
        self._io_alias = None
        fortran_path, alias = _resolve_charmm_fortran_path(
            file_name,
            read_only=read_only,
            append=append,
        )
        self.file_name = fortran_path
        self._io_alias = alias
        self.open()

    def __del__(self):
        """class destructor

        if the file is open, close it
        """
        if self.is_open:
            self.close()

    def __enter__(self):
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self, read_only=None, append=None, formatted=None):
        """Open the file

        Parameters
        ----------
        read_only : bool, optional
            Override the read/write mode for this open call.
            If not given, uses the current setting.
        append : bool, optional
            Override the append mode.  If not given, uses the
            current setting.
        formatted : bool, optional
            Override the formatted flag.  If not given, uses the
            current setting.

        Returns
        -------
        bool
            True if file is open
        """
        if self.is_open:
            return True

        if read_only is not None:
            self.read_only = read_only
        if append is not None:
            self.append = append
        if formatted is not None:
            self.formatted = formatted

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

        fmt = ctypes.c_int(0)
        if self.formatted:
            fmt = ctypes.c_int(1)

        new_unit = ctypes.c_int(self.file_unit)

        # Open the file by calling a charmm fortran routine
        charmm_unit = lib.charmm_file_open(
            fn, len_fn,
            to_read, to_write, to_append,
            fmt,
            new_unit)

        if charmm_unit != -1:
            self.file_unit = charmm_unit
            self.is_open = True

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
            success_bit = lib.charmm_file_close(unit)

        if success_bit == 1:
            self.is_open = False
            if self._io_alias is not None:
                self._io_alias.finalize()

        return not self.is_open
