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

""" Functions to construct cartesian coordinates from internal coordinate values  

There are flexible editing commands for manipulating the data structure. This module, together with the coordinate manipulation commands (see the `coor` module) and the I/O commands (see the `charmm` module), provides model building functionality.  
The internal coordinate data structure can also be used for analysis purposes.  

Corresponds to CHARMM module `intcor`  
See [intcor documentation](https://academiccharmm.org/documentation/version/c47b1/intcor)  

Examples
========
>>> import pycharmm
>>> import pycharmm.generate as gen
>>> >>> import pycharmm.ic as ic
>>> read.sequence_string('ALA')
>>> gen.new_segment(seg_name='ALAD',
...                 first_patch='ACE',
...                 last_patch='CT3',
...                 setup_ic=True)

>>> ic.prm_fill(replace_all=True)
>>> ic.seed(1,'CAY',1,'CY',1,'N')
>>> ic.build()

"""

import ctypes

import pycharmm.lib as lib


# fill internal coords with values from parameter file
# if replace_all is true, all angle and bond vals are filled
# regardless of existing vals
def prm_fill(replace_all):
    """Fill internal coords with values from parameter file

    Parameters
    ----------
    replace_all : bool
                  if true, all angle and bond vals are filled
                  regardless of existing vals

    Returns
    -------
    int
        1 indicates success, any other value indicates failure
    """
    flag = ctypes.c_int(0)
    if replace_all:
        flag = ctypes.c_int(1)

    lib.charmm.ic_fill_from_param_file(ctypes.byref(flag))


def show():
    """Print the internal coordinates of a molecule
    """
    lib.charmm.ic_print()


def edit_dihedral(res1, atom1,
                  res2, atom2,
                  res3, atom3,
                  res4, atom4,
                  new_psi):
    """Replace or create a dihedral angle

    Parameters
    ----------
    res1 : int
           residue number for atom1
    atom1 : str
            atom name
    res2 : int
           residue number for atom2
    atom2 : str
            atom name
    res3 : int
           residue number for atom3
    atom3 : str
            atom name
    res4 : int
           residue number for atom4
    atom4 : str
            atom name
    new_psi : float
              the new angle in degrees between the two planes

    Returns
    -------
    int
       1 indicates success, any other value indicates failure
    """
    res1 = ctypes.c_int(res1)
    atom1 = ctypes.create_string_buffer(atom1.encode())

    res2 = ctypes.c_int(res2)
    atom2 = ctypes.create_string_buffer(atom2.encode())

    res3 = ctypes.c_int(res3)
    atom3 = ctypes.create_string_buffer(atom3.encode())

    res4 = ctypes.c_int(res4)
    atom4 = ctypes.create_string_buffer(atom4.encode())

    new_psi = ctypes.c_double(new_psi)

    lib.charmm.ic_edit_dihedral(ctypes.byref(res1), atom1,
                                ctypes.byref(res2), atom2,
                                ctypes.byref(res3), atom3,
                                ctypes.byref(res4), atom4,
                                ctypes.byref(new_psi))


def edit_angle(res1, atom1,
               res2, atom2,
               res3, atom3,
               new_angle):
    """Replace or create an angle

    Parameters
    ----------
    res1 : int
           residue number for atom1
    atom1 : str
            atom name
    res2 : int
           residue number for atom2
    atom2 : str
            atom name
    res3 : int
           residue number for atom3
    atom3 : str
            atom name
    new_angle : float
                the new angle in degrees

    Returns
    -------
    int
        1 indicates success, any other value indicates failure
    """
    res1 = ctypes.c_int(res1)
    atom1 = ctypes.create_string_buffer(atom1.encode())

    res2 = ctypes.c_int(res2)
    atom2 = ctypes.create_string_buffer(atom2.encode())

    res3 = ctypes.c_int(res3)
    atom3 = ctypes.create_string_buffer(atom3.encode())

    new_angle = ctypes.c_double(new_angle)

    lib.charmm.ic_edit_angle(ctypes.byref(res1), atom1,
                             ctypes.byref(res2), atom2,
                             ctypes.byref(res3), atom3,
                             ctypes.byref(new_angle))


def edit_dist(res1, atom1,
              res2, atom2,
              new_dist):
    """Change the distance between two atoms

    Parameters
    ----------
    res1 : int
           residue number for atom1
    atom1 : str
            atom name
    res2 : int
           residue number for atom2
    atom2 : str
            atom name
    new_dist : float
               the distance between the two atoms

    Returns
    -------
    int
        1 indicates success, any other value indicates failure
    """
    res1 = ctypes.c_int(res1)
    atom1 = ctypes.create_string_buffer(atom1.encode())

    res2 = ctypes.c_int(res2)
    atom2 = ctypes.create_string_buffer(atom2.encode())

    new_dist = ctypes.c_double(new_dist)

    lib.charmm.ic_edit_dist(ctypes.byref(res1), atom1,
                            ctypes.byref(res2), atom2,
                            ctypes.byref(new_dist))


def seed(res1, atom1,
         res2, atom2,
         res3, atom3):
    """Place first three atoms for building reference

    When the cartesian coordinates are not specified for any atoms,
    the `BUILd` command cannot be used to generate positions since all positions
    are determined relative to known positions. The `SEED` command specifies the
    positions of the three atoms. It puts the first at the origin, the second
    on the x-axis, and the third in the xy-plane. The three atoms must have
    entries in the IC file corresponding to: dist 1-2, angle 1-2-3, dist 2-3.

    Parameters
    ----------
    res1, res2, res3 : int
        residue ids of each atom
    atom1, atom2, atom3 : str
        name (type) of each atom 
    """
    res1 = ctypes.c_int(res1)
    atom1 = ctypes.create_string_buffer(atom1.encode())

    res2 = ctypes.c_int(res2)
    atom2 = ctypes.create_string_buffer(atom2.encode())

    res3 = ctypes.c_int(res3)
    atom3 = ctypes.create_string_buffer(atom3.encode())

    lib.charmm.ic_seed(ctypes.byref(res1), atom1,
                       ctypes.byref(res2), atom2,
                       ctypes.byref(res3), atom3)


def build():
    """Determine coordinates for all unspecified atoms

    This command determines coordinates from the data in the IC file (wherever possible).
    The user is responsible for making sure that the designation for all atoms
    is unique. In the case that the system is over specified, an atom is
    placed on the first opportunity (no checking is done for currently placed
    atoms).
    """
    lib.charmm.ic_build()
