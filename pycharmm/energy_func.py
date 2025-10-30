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

"""A class to set or just store a python function to run during an energy calculation 
"""

import ctypes
import pycharmm.lib as lib


class EnergyFunc:
    """A class to set or just store a python function to run during an energy calculation
    """
    def __init__(self, py_func, set_func_now=True):
        """class constructor

        The `py_func` parameter should be a python function that takes 7 arguments:  
        `natoms`: the integer number of elements in the next six arrays  
        `x_pos`, `y_pos`, `z_pos`: double precision floating point arrays of atom positions  
        `dx`, `dy`, `dz`: double precision floating point arrays  

        Inside the `py_func` function, the array arguments must be updated element by element
        since they are passed at C pointers. Slicing on the left hand side of an equal sign
        will not work for updating elements. List comprehensions on the right hand side
        probably will not either.

        Parameters
        ----------
        py_func : callable python function
            a python function to run during energy calculations
        set_func_now : bool, default = True
            just store the function and don't run it during future calcs
        """
        self.func_type = ctypes.CFUNCTYPE(ctypes.c_double,
                                          ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double))
        self.energy_func = self.func_type(py_func)
        self.is_set = False
        if set_func_now:
            self.set_func()

    def __del__(self):
        """class destructor
        """
        self.unset_func()

    def set_func(self, py_func=None):
        """Set the stored function or a new function to run during energy calcs.

        The `py_func` parameter should be a python function that takes 7 arguments:  
        `natoms`: the integer number of elements in the next six arrays  
        `x_pos`, `y_pos`, `z_pos`: double precision floating point arrays of atom positions  
        `dx`, `dy`, `dz`: double precision floating point arrays  

        Inside the `py_func` function, the array arguments must be updated element by element
        since they are passed at C pointers. Slicing on the left hand side of an equal sign
        will not work for updating elements. List comprehensions on the right hand side
        probably will not either.

        Parameters
        ----------
        py_func : callable python function 
            A python function to run during energy calculations
        """
        if py_func:
            self.energy_func = self.func_type(py_func)

        lib.charmm.func_set(self.energy_func)
        self.is_set = True

    def unset_func(self):
        """Just store the function and do not run it during energy calculations
        """
        lib.charmm.func_unset()
        self.is_set = False
