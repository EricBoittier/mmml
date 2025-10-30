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

"""
Set a python function to run during dynamics  
"""

import ctypes
import pycharmm.lib as lib


class CustomDynam:
    """Set a python function to run during dynamics"""
    def __init__(self, py_func, set_func_now=True):
        """Constructor

        The `py_func` parameter should be a python function that takes 11 arguments:  
        `current_step`: the current step number of the dynamics run  
        `natoms`: the integer number of elements in the next six arrays  
        `vx`, `vy`, `vz`: double precision floating point arrays of atom velocities  
        `x_new`, `y_new`, `z_new`: double precision floating point arrays of atom positions  
        `x_old`, `y_old`, `z_old`: double precision floating point arrays of atom positions  

        Inside the py_func function, the array arguments must be updated element by element
        since they are passed at C pointers. Slicing on the left hand side of an equal sign
        will not work for updating elements. List comprehensions on the right hand side
        probably will not either.

        Parameters
        ----------
        py_func : callable  
            the function to run during dynamics
        set_func_now : bool, default = True
            do not run the function now, but store it
        """
        self.func_type = ctypes.CFUNCTYPE(ctypes.c_double,
                                          ctypes.c_int,
                                          ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double))
        self.dynam_func = self.func_type(py_func)
        self.is_set = False
        if set_func_now:
            self.set_func()

    def __del__(self):
        """Destructor"""
        self.unset_func()

    def set_func(self, py_func=None):
        """Set a python function to run during dynamics

        The `py_func` parameter should be a python function that takes 11 arguments:  
        `current_step`: the current step number of the dynamics run  
        `natoms`: the integer number of elements in the next six arrays  
        `vx`, `vy`, `vz`: double precision floating point arrays of atom velocities  
        `x_new`, `y_new`, `z_new`: double precision floating point arrays of atom positions  
        `x_old`, `y_old`, `z_old`: double precision floating point arrays of atom positions  

        Inside the py_func function, the array arguments must be updated element by element
        since they are passed at C pointers. Slicing on the left hand side of an equal sign
        will not work for updating elements. List comprehensions on the right hand side
        probably will not either.

        Parameters
        ----------
        py_func : callable
            the function to run during dynamics
        """
        if py_func:
            self.dynam_func = self.func_type(py_func)

        lib.charmm.custom_dynam_set(self.dynam_func)
        self.is_set = True

    def unset_func(self):
        """Do not run a python function during dynamics"""
        lib.charmm.custom_dynam_unset()
        self.is_set = False
