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

"""Setup image centering

Corresponds to CHARMM command `IMAGE`

See CHARMM documentation [IMAGES](<https://academiccharmm.org/documentation/version/c47b1/images>)
for more information



Functions
=========
- `setup_residue` -- setup image centering for a residue
- `setup_segment` -- setup image centering for a segment

Examples
========
>>> import pycharmm.image as image

Set box center at the origin and center by segment for a segment named PROT 
>>> image.setup_segment(0,0,0,'PROT')

Set box center at the origin and center by residue for water TIP3
>>> image.setup_residue(0,0,0,'TIP3')

"""

import ctypes
import pycharmm.lib as lib


def setup_residue(center_x, center_y, center_z, resname):
    """Setup image centering for a residue

    Parameters
    ----------
    center_x : float
               new center for x coords
    center_y : float
               new center for y coords
    center_z : float
               new center for z coords
    resname : string
              name of residue to center on next image update

    Returns
    -------
    success : integer
              1 == no error
    """
    x = ctypes.c_double(center_x)
    y = ctypes.c_double(center_y)
    z = ctypes.c_double(center_z)
    c_name = ctypes.c_char_p(resname.encode('utf-8'))
    success = lib.charmm.image_setup_residue(ctypes.byref(x),
                                             ctypes.byref(y),
                                             ctypes.byref(z),
                                             c_name)
    return success


def setup_segment(center_x, center_y, center_z, segid):
    """Setup image centering for a segment

    Parameters
    ----------
    center_x : float
               new center for x coords
    center_y : float
               new center for y coords
    center_z : float
               new center for z coords
    segid : string
            name of segment to center on next image update

    Returns
    -------
    success : integer
              1 == no error
    """
    x = ctypes.c_double(center_x)
    y = ctypes.c_double(center_y)
    z = ctypes.c_double(center_z)
    c_name = ctypes.c_char_p(segid.encode('utf-8'))
    success = lib.charmm.image_setup_segment(ctypes.byref(x),
                                             ctypes.byref(y),
                                             ctypes.byref(z),
                                             c_name)
    return success


def setup_selection(center_x, center_y, center_z, selection):
    """Setup image centering for a selection of atoms

    Parameters
    ----------
    center_x : float
               new center for x coords
    center_y : float
               new center for y coords
    center_z : float
               new center for z coords
    selection : pycharmm.SelectAtoms
                selection[i] == True <=> atom i is selected

    Returns
    -------
    success : integer
              1 == no error
    """
    x = ctypes.c_double(center_x)
    y = ctypes.c_double(center_y)
    z = ctypes.c_double(center_z)
    sel_c = selection.as_ctypes()
    success = lib.charmm.image_setup_selection(ctypes.byref(x),
                                               ctypes.byref(y),
                                               ctypes.byref(z),
                                               sel_c)
    return success


# Addition Kai Toepfer May 2022
def get_ucell():
    """Get unit cell edge sizes and angles

    Returns
    -------
    ucell : float
            current unit cell vector x
    """
    
    ucell = (ctypes.c_double * 6)(0)
    status = lib.charmm.image_get_ucell(ucell)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem fetching unit cell data.')

    return list(ucell)


def get_ntrans():
    """Get number of image cells

    Returns
    -------
    ntrans : integer
             number of image cells
    """
    
    ntrans = (ctypes.c_int * 1)()
    status = lib.charmm.image_get_ntrans(ntrans)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem fetching NTRANS.')

    return ntrans[0] + 1


def update_bimag():
    """Update image - primary atoms non-bonded exclusion list
    """
    
    lib.charmm.image_update_bimag()
    
    return
