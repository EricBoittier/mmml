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

"""Build a crystal with any space group symmetry,
   optimise its lattice parameters and
   molecular coordinates and
   carry out a vibrational analysis using the options.

Corresponds to CHARMM command `CRYStal`

See CHARMM documentation [crystl](<https://academiccharmm.org/documentation/version/c47b1/crystl>)
for more information

Functions
=========
- `define_cubic` -- defines a cubic lattice for a new crystal
- `define_tetra` -- defines a tetragonal lattice for a new crystal
- `define_ortho` -- defines a orthorhombic lattice for a new crystal
- `define_mono` -- defines a monoclinic lattice for a new crystal
- `define_tri` -- defines a triclinic lattice for a new crystal
- `define_hexa` -- defines a hexagonal lattice for a new crystal
- `define_rhombo` -- defines a rhombohedral lattice for a new crystal
- `define_octa` -- defines a octahedral lattice for a new crystal
- `define_rhdo` -- defines a rhombic dodecahedron lattice for a new crystal

Examples
========
Build a cubic simulation box of 0.7 nm 
>>> crystal.define_cubic(70)
>>> crystal.build(12)
"""

import ctypes
import pycharmm.lib as lib


def define_cubic(length):
    """Defines a cubic lattice and constants for a new crystal

    Parameters
    ----------
    length : float
        length of all sides

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length = ctypes.c_double(length)
    success = lib.charmm.crystal_define_cubic(ctypes.byref(length))
    return success


def define_tetra(length_a, length_c):
    """Defines a tetragonal lattice and constants for a new crystal

    The alpha, beta and gamma angles are all 90.0 degrees.
    The length of sides a and b are equal.

    Parameters
    ----------
    length_a : float
        length of sides a and b
    length_c : float
        length of side c

    Returns
    -------
    bool 
        True for success, otherwise False
    """
    length_a = ctypes.c_double(length_a)
    length_c = ctypes.c_double(length_c)
    success = lib.charmm.crystal_define_tetra(ctypes.byref(length_a),
                                              ctypes.byref(length_c))
    return success


def define_ortho(length_a, length_b, length_c):
    """Defines a orthorhombic lattice and constants for a new crystal

    The alpha, beta and gamma angles are all 90.0 degrees.

    Parameters
    ----------
    length_a : float 
        length of side a
    length_b : 
        float length of side b
    length_c : float 
        length of side c

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    success = lib.charmm.crystal_define_ortho(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c))
    return success


def define_mono(length_a, length_b, length_c, angle_beta):
    """Defines a monoclinic lattice and constants for a new crystal

    The alpha and gamma angles are both 90.0 degrees.

    Parameters
    ----------
    length_a : float 
        length of side a
    length_b : float
        length of side b
    length_c : float
        length of side c
    angle_beta : float 
        measure of angle beta in degrees

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    angle_beta = ctypes.c_double(angle_beta)
    success = lib.charmm.crystal_define_ortho(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c),
                                              ctypes.byref(angle_beta))
    return success


def define_tri(length_a, length_b, length_c,
               angle_alpha, angle_beta, angle_gamma):
    """Defines a triclinic lattice and constants for a new crystal

    Parameters
    ----------
    length_a : float
        length of side a
    length_b : float 
        length of side b
    length_c : float 
        length of side c
    angle_alpha : float 
        measure of angle alpha in degrees
    angle_beta : float 
        measure of beta alpha in degrees
    angle_gamma : float 
        measure of gamma alpha in degrees

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    angle_alpha = ctypes.c_double(angle_alpha)
    angle_beta = ctypes.c_double(angle_beta)
    angle_gamma = ctypes.c_double(angle_gamma)
    success = lib.charmm.crystal_define_ortho(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c),
                                              ctypes.byref(angle_alpha),
                                              ctypes.byref(angle_beta),
                                              ctypes.byref(angle_gamma))
    return success


def define_hexa(length_a, length_c):
    """Defines a hexagonal lattice and constants for a new crystal

    Parameters
    ----------
    length_a : float 
        lengths of sides a and b
    length_c : float
        length of side c

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length_a = ctypes.c_double(length_a)
    length_c = ctypes.c_double(length_c)
    success = lib.charmm.crystal_define_hexa(ctypes.byref(length_a),
                                             ctypes.byref(length_c))
    return success


def define_rhombo(length, angle):
    """Defines a rhombohedral lattice and constants for a new crystal

    Parameters
    ----------
    length : float
        length of each side
    angle : float 
        measure of each angle in degrees, must be between 0 and 120

    Returns
    -------
    bool
        True for success, otherwise False
    """
    if angle <= 0.0 or angle >= 120.0:
        raise ValueError("Value %d out of range (0.0, 120.0)" % (angle,))

    length = ctypes.c_double(length)
    angle = ctypes.c_double(angle)
    success = lib.charmm.crystal_define_rhombo(ctypes.byref(length),
                                               ctypes.byref(angle))
    return success


def define_octa(length):
    """Defines a octahedral lattice and constants for a new crystal

    Parameters
    ----------
    length : float
        the length of each side

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length = ctypes.c_double(length)
    success = lib.charmm.crystal_define_octa(ctypes.byref(length))
    return success


def define_rhdo(length):
    """Defines a rhombic dodecahedron lattice and constants for a new crystal

    Parameters
    ----------
    length: float
        the length of each side

    Returns
    -------
    bool
        True for success, otherwise False
    """
    length = ctypes.c_double(length)
    success = lib.charmm.crystal_define_rhdo(ctypes.byref(length))
    return success


def build(cutoff, sym_ops=None):
    """Build the crystal by repeatedly appling specified transformations

    Parameters
    ----------
    cutoff : float 
        images within cutoff distance are included in transformation list
    sym_ops : list[string] 
        transformations in (X, Y, Z) format

    Returns
    -------
    bool
        True for success, otherwise False
    """
    if sym_ops is None:
        sym_ops = []

    nops = len(sym_ops)

    str_array = (ctypes.c_char_p * nops)()
    for i, sym_op in enumerate(sym_ops):
        str_array[i] = sym_op.encode('utf-8')

    cutoff = ctypes.c_double(cutoff)
    nops = ctypes.c_int(nops)

    success = lib.charmm.crystal_build(ctypes.byref(cutoff),
                                       ctypes.byref(str_array),
                                       ctypes.byref(nops))
    return success
