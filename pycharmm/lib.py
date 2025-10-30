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

"""Finds and loads the CHARMM shared library

On importing pycharmm, this module looks for an environment variable named
`CHARMM_HOME` to find path to CHARMM shared library. 
The extension for the library is set
depending on the output of platform.system
"""


import ctypes
import os
import os.path
import platform


class CharmmLib:
    def __init__(self, charmm_lib_dir=''):
        self.charmm_lib_name = 'libcharmm'
        if charmm_lib_dir:
            self.charmm_lib_name = os.path.join(charmm_lib_dir,
                                                self.charmm_lib_name)

        sys_name = platform.system()
        if sys_name == 'Darwin':
            self.charmm_lib_name += '.dylib'
        elif sys_name == 'Linux':
            self.charmm_lib_name += '.so'
        elif sys_name == 'Windows':
            self.charmm_lib_name += '.dll'

        self.lib = None
        self.init_charmm()

        self.dlclose = ctypes.CDLL(None).dlclose  # does not work
        self.dlclose.argtypes = [ctypes.c_void_p]

    def __del__(self):
        self.del_charmm()


    def init_charmm(self):
        self.lib = ctypes.CDLL(self.charmm_lib_name)
        self.lib.init_charmm()

    def del_charmm(self):
        self.lib.del_charmm()  # initiates 'normal stop'
        # does not work
        # self.lib.dlclose(self.lib)


charmm_lib = CharmmLib(os.environ.get('CHARMM_LIB_DIR', ''))
charmm = charmm_lib.lib
