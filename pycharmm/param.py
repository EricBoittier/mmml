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

"""Get data from CHARMM associated with parameter files

"""

import ctypes
import pycharmm.lib as lib
import pycharmm.coor as coor


def get_natc():
    """Get the size of the atc array (see `get_atc()`)

    Returns
    -------
    n : integer
        current total number of atc entries
    """
    n = lib.charmm.param_get_natc()
    return n


def get_atc():
    """Export a copy of the atom type 'CHEM' codes/names

    Returns
    -------
    list[str]
          CHEM name for atom type code
    """
    n = get_natc()
    buffers = [ctypes.create_string_buffer(8) for _ in range(n)]
    pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof, buffers))

    lib.charmm.param_get_atc(pointers)
    res = [b.value.decode(errors='ignore') for b in buffers]
    return res

def get_charge() :
    #Created by Yujin Wu, wyujin@umich.edu
    """Gets the charge of the atoms in the simulation
    
    Returns
    -------
    list[float]
       Charge of all the atoms.
    """
    natom = coor.get_natom()
    c_charge = (ctypes.c_double * natom)()
    lib.charmm.param_get_charge(c_charge)

    charge = [c_charge[i] for i in range(natom)]
    return charge

def get_vdwr() :
    #Created by Yujin Wu, wyujin@umich.edu
    """Gets the vdW radius (vdwr) of the atoms in the simulation

    Returns
    -------
    list[float]
        vdW radius of all the atoms.
    """
    natom = coor.get_natom()
    c_vdwr = (ctypes.c_double * natom)()
    lib.charmm.param_get_vdwr(c_vdwr)

    vdwr = [c_vdwr[i] for i in range(natom)]
    return vdwr

def get_epsilon():
    #Created by Yujin Wu, wyujin@umich.edu
    """Gets the epsilon of the atoms in the simulation

    Returns
    -------
    list[float]
        vdW ebergy well depth (epsilon) for all the atoms.
    """
    natom = coor.get_natom()
    c_epsilon = (ctypes.c_double * natom)()
    lib.charmm.param_get_epsilon(c_epsilon)

    epsilon = [c_epsilon[i] for i in range(natom)]
    return epsilon
