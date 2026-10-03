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

"""Functions for manipulating coordinates of atoms.

Corresponds to CHARMM module `corman` for Coordinate manipulation and analyses.
See [corman documentation](https://academiccharmm.org/documentation/version/c47b1/corman)

"""

import ctypes

import pandas
import numpy as np

import pycharmm
from pycharmm.loader import lib
from pycharmm.trace import traced

def get_natom():
    """Returns the number of atoms currently in the simulation

    Returns
    -------
    natom : int
        number of atoms currently in the simulation
    """
    natom = lib.coor_get_natom()
    return natom


def _ptr_double(arr: np.ndarray):
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )


def set_positions_array(x, y, z):
    """Sets atom positions directly from array-like 1D coordinates."""
    natom = get_natom()
    return lib.coor_set_positions(
        _ptr_double(x[:natom]),
        _ptr_double(y[:natom]),
        _ptr_double(z[:natom]),
    )


def get_positions_array():
    """Gets atom positions directly into NumPy float64 arrays."""
    natom = get_natom()
    x = np.empty(natom, dtype=np.float64)
    y = np.empty(natom, dtype=np.float64)
    z = np.empty(natom, dtype=np.float64)
    lib.coor_get_positions(_ptr_double(x), _ptr_double(y), _ptr_double(z))
    return x, y, z


def set_forces_array(dx, dy, dz):
    """Sets atom forces directly from array-like 1D gradients."""
    natom = get_natom()
    return lib.coor_set_forces(
        _ptr_double(dx[:natom]),
        _ptr_double(dy[:natom]),
        _ptr_double(dz[:natom]),
    )


def get_forces_array():
    """Gets atom forces directly into NumPy float64 arrays."""
    natom = get_natom()
    dx = np.empty(natom, dtype=np.float64)
    dy = np.empty(natom, dtype=np.float64)
    dz = np.empty(natom, dtype=np.float64)
    lib.coor_get_forces(_ptr_double(dx), _ptr_double(dy), _ptr_double(dz))
    return dx, dy, dz


@traced('coor.set_positions')
def set_positions(pos):
    """Sets the positions of the atoms in the simulation

    Parameters
    ----------
    pos : pandas.core.frame.DataFrame
        a dataframe with columns named x, y and z

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0 and pos.empty: # Or just natom == 0
        return 0
    if len(pos) < natom:
        raise ValueError(f"Input DataFrame has {len(pos)} rows, but {natom} atoms exist. Provide coordinates for all atoms.")

    # Ensure correct dtype and contiguity from DataFrame columns
    # Slicing with .iloc[0:natom] ensures we only take necessary data
    pos_x_np = pos['x'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    pos_y_np = pos['y'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    pos_z_np = pos['z'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)

    c_x = (ctypes.c_double * natom)(*pos_x_np)
    c_y = (ctypes.c_double * natom)(*pos_y_np)
    c_z = (ctypes.c_double * natom)(*pos_z_np)

    natom_set = lib.coor_set_positions(c_x, c_y, c_z)
    return natom_set # Return what the C function returns (usually natom or status)


def get_positions():
    """Gets the positions of the atoms in the simulation

    Returns
    -------
    pos : pandas.core.frame.DataFrame
        a dataframe with columns named *x*, *y* and *z*
    """
    natom = get_natom()
    if natom == 0:
        return pandas.DataFrame({'x': [], 'y': [], 'z': []}) # Return empty DataFrame

    c_x = (ctypes.c_double * natom)()
    c_y = (ctypes.c_double * natom)()
    c_z = (ctypes.c_double * natom)()

    lib.coor_get_positions(c_x, c_y, c_z)

    # Convert ctypes arrays directly to numpy arrays, then to DataFrame
    # Making a copy is important as as_array shares memory by default.
    pos_x_np = np.ctypeslib.as_array(c_x).copy()
    pos_y_np = np.ctypeslib.as_array(c_y).copy()
    pos_z_np = np.ctypeslib.as_array(c_z).copy()

    pos = pandas.DataFrame({'x': pos_x_np, 'y': pos_y_np, 'z': pos_z_np})
    return pos


@traced('coor.set_weights')
def set_weights(weights):
    """Sets the weights of the atoms in the simulation

    Parameters
    ----------
    weights : list[float]
        list of one weight for each atom 0:natom

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0:
        return 0
    if len(weights) < natom:
        raise ValueError(f"Input 'weights' list has {len(weights)} elements, but {natom} atoms exist. Provide weights for all atoms.")

    # Convert list to NumPy array, then to ctypes
    weights_np = np.array(weights[0:natom], dtype=np.double)
    c_weights = (ctypes.c_double * natom)(*weights_np)

    natom_set = lib.coor_set_weights(c_weights)
    return natom_set # Return what the C function returns


def get_weights():
    """Gets the weights of the atoms in the simulation

    Returns
    -------
    weights : list[float]
        a list of float atom weights 0:natom
    """
    natom = get_natom()
    if natom == 0:
        return [] # Return empty list

    c_weights = (ctypes.c_double * natom)()
    lib.coor_get_weights(c_weights)

    # Convert ctypes array to NumPy array, then to list
    weights_np = np.ctypeslib.as_array(c_weights).copy()
    return weights_np.tolist()  # Return type is list[float]


@traced('coor.copy_forces')
def copy_forces(mass=False, selection=None):
    """Copy the forces to the comparison set

    Parameters
    ----------
    mass : bool, default = False
        Whether to weight the forces by mass
    selection : pycharmm.SelectAtoms
        only copy forces of selected atoms

    Returns
    -------
    bool
        true if successful
    """
    if not selection:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()
    c_mass = ctypes.c_int(mass)

    status = lib.coor_copy_forces(ctypes.byref(c_mass), c_sel)

    status = bool(status)
    return status


@traced('coor.set_forces')
def set_forces(forces):
    """Sets the forces of the atoms

    Parameters
    ----------
    forces : pandas.core.frame.DataFrame
        a dataframe with columns named dx, dy and dz

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0 and forces.empty:
        return 0
    if len(forces) < natom:
        raise ValueError(f"Input DataFrame 'forces' has {len(forces)} rows, but {natom} atoms exist. Provide forces for all atoms.")

    dx_np = forces['dx'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    dy_np = forces['dy'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    dz_np = forces['dz'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)

    c_dx = (ctypes.c_double * natom)(*dx_np)
    c_dy = (ctypes.c_double * natom)(*dy_np)
    c_dz = (ctypes.c_double * natom)(*dz_np)

    natom_set = lib.coor_set_forces(c_dx, c_dy, c_dz)
    return natom_set # Return what the C function returns


def get_forces():
    """Gets the forces of the atoms

    Returns
    -------
    forces : pandas.core.frame.DataFrame
        a dataframe with columns named *dx*, *dy* and *dz*
    """
    natom = get_natom()
    if natom == 0:
        return pandas.DataFrame({'dx': [], 'dy': [], 'dz': []})

    c_dx = (ctypes.c_double * natom)()
    c_dy = (ctypes.c_double * natom)()
    c_dz = (ctypes.c_double * natom)()

    lib.coor_get_forces(c_dx, c_dy, c_dz)

    dx_np = np.ctypeslib.as_array(c_dx).copy()
    dy_np = np.ctypeslib.as_array(c_dy).copy()
    dz_np = np.ctypeslib.as_array(c_dz).copy()

    forces = pandas.DataFrame({'dx': dx_np, 'dy': dy_np, 'dz': dz_np})
    return forces


@traced('coor.set_comparison')
def set_comparison(pos):
    """Set the comparison set in CHARMM

    Parameters
    ----------
    pos : pandas.core.frame.DataFrame
        a dataframe with columns named *x*, *y*, *z*, and *w*

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0 and pos.empty:
        return 0
    if len(pos) < natom:
        raise ValueError(f"Input DataFrame 'pos' for comparison set has {len(pos)} rows, but {natom} atoms exist. Provide data for all atoms.")

    x_np = pos['x'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    y_np = pos['y'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    z_np = pos['z'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    w_np = pos['w'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)

    c_x = (ctypes.c_double * natom)(*x_np)
    c_y = (ctypes.c_double * natom)(*y_np)
    c_z = (ctypes.c_double * natom)(*z_np)
    c_w = (ctypes.c_double * natom)(*w_np)

    natom_set = lib.coor_set_comparison(c_x, c_y, c_z, c_w)
    return natom_set # Return what the C function returns


def get_comparison():
    """Gets the comparison of the atoms

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*
    """
    natom = get_natom()
    if natom == 0:
        return pandas.DataFrame({'x': [], 'y': [], 'z': [], 'w': []})

    c_x = (ctypes.c_double * natom)()
    c_y = (ctypes.c_double * natom)()
    c_z = (ctypes.c_double * natom)()
    c_w = (ctypes.c_double * natom)()

    lib.coor_get_comparison(c_x, c_y, c_z, c_w)

    x_np = np.ctypeslib.as_array(c_x).copy()
    y_np = np.ctypeslib.as_array(c_y).copy()
    z_np = np.ctypeslib.as_array(c_z).copy()
    w_np = np.ctypeslib.as_array(c_w).copy()

    comp_set = pandas.DataFrame({'x': x_np, 'y': y_np, 'z': z_np, 'w': w_np})
    return comp_set


@traced('coor.set_comp2')
def set_comp2(pos):
    """Sets the comparison set 2

    Parameters
    ----------
    pos : pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0 and pos.empty:
        return 0
    if len(pos) < natom:
        raise ValueError(f"Input DataFrame 'pos' for comp2 set has {len(pos)} rows, but {natom} atoms exist. Provide data for all atoms.")

    x_np = pos['x'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    y_np = pos['y'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    z_np = pos['z'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    w_np = pos['w'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)

    c_x = (ctypes.c_double * natom)(*x_np)
    c_y = (ctypes.c_double * natom)(*y_np)
    c_z = (ctypes.c_double * natom)(*z_np)
    c_w = (ctypes.c_double * natom)(*w_np)

    natom_set = lib.coor_set_comp2(c_x, c_y, c_z, c_w)
    return natom_set


def get_comp2():
    """Gets the comparison set 2

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*
    """
    natom = get_natom()
    if natom == 0:
        return pandas.DataFrame({'x': [], 'y': [], 'z': [], 'w': []})

    c_x = (ctypes.c_double * natom)()
    c_y = (ctypes.c_double * natom)()
    c_z = (ctypes.c_double * natom)()
    c_w = (ctypes.c_double * natom)()

    lib.coor_get_comp2(c_x, c_y, c_z, c_w)

    x_np = np.ctypeslib.as_array(c_x).copy()
    y_np = np.ctypeslib.as_array(c_y).copy()
    z_np = np.ctypeslib.as_array(c_z).copy()
    w_np = np.ctypeslib.as_array(c_w).copy()

    comp2_set = pandas.DataFrame({'x': x_np, 'y': y_np, 'z': z_np, 'w': w_np})
    return comp2_set


@traced('coor.set_main')
def set_main(pos):
    """Sets the main coordinate set

    Parameters
    ----------
    pos : pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*

    Returns
    -------
    natom : int
        number of atoms in the simulation
    """
    natom = get_natom()
    if natom == 0 and pos.empty:
        return 0
    if len(pos) < natom:
        raise ValueError(f"Input DataFrame 'pos' for main set has {len(pos)} rows, but {natom} atoms exist. Provide data for all atoms.")

    x_np = pos['x'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    y_np = pos['y'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    z_np = pos['z'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)
    w_np = pos['w'].iloc[0:natom].to_numpy(dtype=np.double, copy=True)

    c_x = (ctypes.c_double * natom)(*x_np)
    c_y = (ctypes.c_double * natom)(*y_np)
    c_z = (ctypes.c_double * natom)(*z_np)
    c_w = (ctypes.c_double * natom)(*w_np)

    natom_set = lib.coor_set_main(c_x, c_y, c_z, c_w)
    return natom_set


def get_main():
    """Gets the main coordinates of the atoms

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe with columns named *x*, *y*, *z*, and *w*
    """
    natom = get_natom()
    if natom == 0:
        return pandas.DataFrame({'x': [], 'y': [], 'z': [], 'w': []})

    c_x = (ctypes.c_double * natom)()
    c_y = (ctypes.c_double * natom)()
    c_z = (ctypes.c_double * natom)()
    c_w = (ctypes.c_double * natom)()

    lib.coor_get_main(c_x, c_y, c_z, c_w)

    x_np = np.ctypeslib.as_array(c_x).copy()
    y_np = np.ctypeslib.as_array(c_y).copy()
    z_np = np.ctypeslib.as_array(c_z).copy()
    w_np = np.ctypeslib.as_array(c_w).copy()

    main_set = pandas.DataFrame({'x': x_np, 'y': y_np, 'z': z_np, 'w': w_np})
    return main_set


def hbuild(**kwargs):
    """The HBUILD command adds missing hydrogen atoms to structures

    HBUILD     [atom-selection] hbond-spec  non-bond-spec

               [PHIStp real] [PRINt]  [CUTWater real]

               [WARN] [DISTof real] [ANGLon real]
    Parameters
    ----------
    **kwargs: dictionary
        phistep = [real]
        cutwater = [real]
        distoff  = [real]
        angleon  = [real]
        print    = [bool]
        warn     = [bool]
    """
    if 'selection' not in kwargs.keys():
        kwargs['selection'] = pycharmm.SelectAtoms(hydrogens=True,lonepairs=True,initials=False)
    hbuild_script = pycharmm.script.CommandScript('hbuild',**kwargs)
    hbuild_script.run()
    return

def convert(**kwargs):
    """The COOR CONVert command will cause the coordinates of all
       defined and selected atoms to be transformed from the unit cell to
       cartesian coordinates or back from cartesian to fractional coordinates.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor CONVert CHARMM command

        FROM: [string]  [FRAC|ALIG|SYMM]

        TO:   [string]  [FRAC|ALIG|SYMM]

        COMP: [True/False] operates on COMP/MAIN

        selection: SelectAtoms()

        Two orientations in cartesian coordinates are supported :

       ALIGned:   - in which b-vector is along y-axis and a-vector
                    in xy-plane (this is old charmm standard)

       SYMMetric:  - in which shape matrix constructed from unit
                     cell vectors is symmetric

       Two keywords in any order [FRAC|ALIG|SYMM] are required after CONVert.

       Unit cell parameters (a,b,c,alpha,beta,gamma) follow in the same line.

       The angle values are specified in degrees. See the routine CONCOR for
       details concerning the transformation.


       CONVert-from/to-unit-cell [ from | to ] -

                 [atom-selection] [COMP] [IMAGe] -

                 a  b  c   alpha   beta  gamma

                 [ from | to ] ::= [ FRACtional | SYMMetric | ALIGned ]
    """
    xucell = {}
    for p in ['XTLA','XTLB','XTLC','XTLALPHA','XTLBETA','XTLGAMMA']:
        xucell[p] = pycharmm.lingo.get_energy_value(p)
    if 'a' not in kwargs.keys(): kwargs['a'] = xucell['XTLA']
    if 'b' not in kwargs.keys(): kwargs['b'] = xucell['XTLB']
    if 'c' not in kwargs.keys(): kwargs['c'] = xucell['XTLC']
    if 'alpha' not in kwargs.keys(): kwargs['alpha'] = xucell['XTLALPHA']
    if 'beta' not in kwargs.keys(): kwargs['beta'] = xucell['XTLBETA']
    if 'gamma' not in kwargs.keys(): kwargs['gamma'] = xucell['XTLGAMMA']
    newdict = {}
    cmd = ''
    for k, v in kwargs.items():
        if 'from' in k.lower():
            newdict[v] = True

        elif 'to' in k.lower():
            newdict[v] = True
        elif k == 'a' or k == 'b' or k == 'c' or k == 'alpha' or\
             k == 'beta' or k == 'gamma':
            cmd += f'{v} '
        else:
            newdict[k]=v
    newdict[cmd] = True

    convert_script = pycharmm.script.CommandScript('coor',convert=True,**newdict)

    convert_script.run()
    return

def orient(**kwargs):
    """Modifies coordinates of all atoms according to the passed flags

    The select set of atoms is first centered about the origin,
    and then rotated to either align with the axis,
    or the other coordinate set.

    The *RMS* keyword will use the other coordinate set as a rotation reference.

    The *MASS* keyword cause a mass weighting to be done. This will
    align the specified atoms along their moments of inertia. When the RMS
    keyword is not used, then the structure is rotated so that its principle
    geometric axis coincides with the X-axis and the next largest coincides
    with the Y-axis. This command is primarily used for preparing a
    structure for graphics and viewing. It can also be used for finding
    RMS differences, and in conjunction with the vibrational analysis.

    The *NORO*tation keyword will suppress rotations. In this case,
    only one coordinate set will be modified.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor orient CHARMM command
    """
    oscript = pycharmm.script.CommandScript('coor', orient=True, **kwargs)
    oscript.run()

    orie = {}
    for p in ['XAXI','YAXI','ZAXI','RAXI','XCEN','YCEN','ZCEN',
              'RMS', 'THET', 'XMOV','YMOV','ZMOV']:
        val = pycharmm.lingo.get_energy_value(p, default=None)
        if val is not None:
            orie[p] = val
    return orie


def volume(**kwargs):
    """Executes the coor volume command

    Input
    =====
    space: int - number of pixels to use in computing volume
    selection: AtomSelection() - selection of atoms for which to compute the volume

    Output
    ======
    volume real - volume of selected atoms in cubic Angstroms

    The VOLUme command will compute the volume of a selected set of
    atoms.  Its operation is the same as that of the SEARch command, except
    that only the volume is printed and the degree of exposure for each atom
    is returned in the weighting array.  The SCALAR storage arrays must be filled
    before using this command.  The first storage array [1] must contain
    the radii of each atom (RMIN) and the second storage array must contain the
    outer probe distance (RMAX) for each atom.  The free volume within the RMIN
    to RMAX range and not within RMIN of any other atom will be returned in the
    weighting array as a ratio of the maximum possible value.  For example a
    completely exposed atom will return a value of 1.0 and an atom in the interior
    of a protein would return a value of 0.0.  The HOLEs keyword feature
    causes holes within the selected atoms to be filled before computing
    the total volume and the accesible volume.

    SPACE is a maximum number of cubic pixels
    i.e. SPACE = x_points * y_points * z_points
    Larger SPACE value results in more accurate calculation but it takes more
    memory an computer time.

    Number of points in x,y and z directions are
    determined according to the formula:

    factor = ( SPACE / (a*b*c) ) ** (1/3)

    x_points = factor*a

    y_points = factor*b

    z_points = factor*c

    where a, b and c are dimensions of the smallest rectangular box
    enclosing the molecule.
    """
    volume_command = pycharmm.script.CommandScript('coor', volume=True, **kwargs)
    volume_command.run()
    builtins = pycharmm.lingo.get_charmm_builtins()
    vol_dict = {}
    for item in ['NVAC','NSEL','VOLUME','VOLU','FREEVOL']:
        vol_dict[item] = builtins[item]
    return vol_dict


def diff(**kwargs):
    """Calculates the differences between coordinate sets

    The DIFF command will compute the differences between the main
    and comparison set (or the reverse) and store this difference in the
    modified coordinate set. Undefined or unselected atoms result in a zero.

    If the WEIGht keyword is invoked, then the WCOMP array is subtracted from
    WMAIN and the coordinates are untouched.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor DIFF CHARMM command

        COMP [True/False] put difference in COMP/MAIN

        WEIGht [True/False]

        selection = SelectAtoms()
    """
    dscript = pycharmm.script.CommandScript('coor', diff=True, **kwargs)
    dscript.run()

def rms(**kwargs):
    """The RMS command will compute the RMS or mass weighted RMS
       coordinate differences between the selected set of atoms just as they
       lie. This differences from the COOR ORIENT RMS command in that no coordinate
       modifications are made and no translation is done.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor DIFF CHARMM command
        COMP [True/False] put difference in COMP/MAIN
        WEIGht [True/False]
        selection = SelectAtoms()
    """
    rms_script = pycharmm.script.CommandScript('coor', rms=True, **kwargs)
    rms_script.run()

    return pycharmm.lingo.get_energy_value('RMS')


def translate(**kwargs):
    """The TRANslate command will translate atomic coordinate

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor TRANslate CHARMM command

        XDIR [real] direction/magnitude of translation in X-direction

        YDIR [real] direction/magnitude of translation in Y-direction

        ZDIR [real] direction/magnitude of translation in Z-direction

        DISTance [real] if distance is not set translation is by [xdir,ydir,zdir]

                        if set then [xdir, ydir, zdir] give direction and magnitude is distance

        AXIS [True/False] if true axis from previous axis command is used

        COMP [True/False] operates on COMP/MAIN

        selection = SelectAtoms()
    """
    trans_script = pycharmm.script.CommandScript('coor', trans=True, **kwargs)
    trans_script.run()

def initialize(**kwargs):
    """ The INITialize command returns the coordinate values of the
        specified atoms to their start up values (9999.0). The main use of
        this command is in connection with the IC BUILD command, which may
        only find coordinates for atoms with the initial value.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor INITialize CHARMM command

        COMP [True/False] operates on COMP/MAIN

        selection = SelectAtoms()
    """
    init_script = pycharmm.script.CommandScript('coor', initialize=True, **kwargs)
    init_script.run()

def copy(**kwargs):
    """The COPY command will copy the coordinate values into the
       specified set FROM the other coordinate set.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor COPY CHARMM command

        COMP [True/False] operates on COMP/MAIN

        selection = SelectAtoms()
    """
    copy_script = pycharmm.script.CommandScript('coor', copy=True, **kwargs)
    copy_script.run()

def swap(**kwargs):
    """The SWAP command will cause the coordinate values of the
       specified atoms to be swapped with the comparison set.

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor SWAP CHARMM command

        selection = SelectAtoms()
    """
    swap_script = pycharmm.script.CommandScript('coor', swap=True, **kwargs)
    swap_script.run()

def rotate(**kwargs):
    """The ROTAte command will rotate atomic coordinate

    Parameters
    ----------
    **kwargs : dictionary
        key words for the coor TRANslate CHARMM command

        XDIR [real] X direction of rotation

        YDIR [real] Y direction of rotation

        ZDIR [real] Z direction of rotation

        PHI  [real] angle of rotation about specified axis

        AXIS [True/False] if true axis from previous axis command is used

        MATRIX [ [u(0,0), u(0,1), u(0,2)],
                 [u(1,0), u(1,1), u(1,2)],
                 [u(2,0), u(021), u(2,2)] ]

        COMP [True/False] operates on COMP/MAIN

        selection = SelectAtoms()
    """
    newdict = {}
    m_cmd = None  # Use None instead of False for type consistency
    for k, v in kwargs.items():
        if 'matrix' in k.lower() or k.lower() in 'matrix':
            m_cmd = ''
            for i in range(len(v)):
                m_cmd += f'\n{v[i][0]} {v[i][1]} {v[i][2]}'
        else:
            newdict[k] = v

    # Only pass matrix if it was provided
    if m_cmd is not None:
        rotate_script = pycharmm.script.CommandScript('coor',
                                                      rotate=True,
                                                      matrix=m_cmd,
                                                      **newdict)
    else:
        rotate_script = pycharmm.script.CommandScript('coor',
                                                      rotate=True,
                                                      **newdict)
    rotate_script.run()


def show_comp():
    """Print the comparison set of the atoms


    Returns
    -------
    bool
        true if successful
    """
    status = lib.coor_print_comp()
    status = bool(status)
    return status


def show(**kwargs):
    """Print the main coordinate set of the atoms

    Corresponds to charmm's 'print coord' command
    """
    copy_script = pycharmm.script.CommandScript('print coord', **kwargs)
    copy_script.run()


def stat(selection=None, comp=False, mass=False):
    """Computes *max*, *min*, *ave* for `x`, `y`, `z`, `w` over selection of main or comp sets

    Parameters
    ----------
    selection : pycharmm.SelectAtoms
        a selection of atom indexes to use for stats
    comp : bool
        if true, stats computed for comparison set selection
    mass : bool
        if true, will place the average values at the center of mass

    Returns
    -------
    dict
        A python dictionary with keys
        xmin, xmax, xave,
        ymin, ymax, yave,
        zmin, zmax, zave,
        wmin, wmax, wave,
        n_selected, n_misses
    """
    c_comp = ctypes.c_int(comp)
    c_mass = ctypes.c_int(mass)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    n_stats = 13
    max_label = 5

    vals = (ctypes.c_double * n_stats)()

    labels_bufs = [ctypes.create_string_buffer(max_label)
                   for _ in range(n_stats)]
    labels_ptrs = (ctypes.c_char_p * n_stats)(*map(ctypes.addressof,
                                                   labels_bufs))

    n_sel = ctypes.c_int(0)
    n_misses = ctypes.c_int(0)

    lib.coor_stat(c_sel,
                         ctypes.byref(c_comp), ctypes.byref(c_mass),
                         labels_ptrs, vals,
                         ctypes.byref(n_sel), ctypes.byref(n_misses))

    labels_str = [label.value.decode(errors='ignore')
                  for label in labels_bufs[0:n_stats]]
    labels = [label.strip().lower() for label in labels_str]
    labels = [label for label in labels if label]

    stats = dict(zip(labels, vals))
    stats['n_selected'] = n_sel.value
    stats['n_misses'] = n_misses.value
    return stats

# not compat with python 3.7
# CoordSet = typing.Literal['main', 'comp', 'comp2']
def qexclt(i, j):
    """

    Returns a integer label based on the presence of
    atom with index `i` and `j` in the exclusion list.

    *For dev or internal use only*
    Parameters
    ----------
    i, j: int
          Atom indices

    Returns
    -------
    is14exclt: int
              A value of 1, 2 or 3 is returned
              based on what exclusion list the input pair is in.
    """

    # Converting 0-indexed i, j to 1-indexed values
    # that is compatible with CHARMM arrays
    i = int(i) + 1
    j = int(j) + 1

    # api_coor.F90: integer(c_int), VALUE :: in_i, in_j
    lib.qexclt.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qexclt.restype = ctypes.c_int

    return int(lib.qexclt(ctypes.c_int(i), ctypes.c_int(j)))


def dist(selection1, selection2 = None, cutoff = None, resi = True, omit_14excl = True, omit_nonbonds = False, omit_excl = True ):
    """
    Performs equivalent of CHARMM's `COOR DIST` operation. It can find distances
    between atoms within a selection or find distances between atoms in two selections.
    Returns `None` if no contacts (within specified cutoff value) is found.

    Parameters
    ----------
    selection1: pycharmm.Selection
    selection2: pycharmm.Selection, default = None
                If nothing is eplicitly provided, selection1 is copied into it.
    cutoff: float
            Determine contacts within this value (in Angstroms)
    resi : bool, default = True
           Calculate residue-to-residue minimum distance between two selections.
    omit_14excl: bool, default True
                 Same as `NO14exclusions` in COOR DIST command
    omit_nonbonds: bool, default False
                   Same as `NONOnbonds` in COOR DIST command
    omit_excl: bool ,default True
               Same as `NOEXclusions` in COOR DIST command


    Return
    ------
    contacts : pandas.DataFrame
               If no contacts are found, returns `None`. Otherwise a dataframe whose
               columns are named - 'atomid_1','segid_1','resn_1','resid_1','type_1',
               'atom_2', 'segid_2', 'resn_2','resid_2','type_2', 'distance'
    """
    contactDict = {}

    from pycharmm import atom_info
    from copy import copy
    # import pycharmm.select as pyc_select # Not directly used in this version of dist

    # handle faulty inputs
    if cutoff is None:
        raise ValueError("FATAL ERROR: cutoff value unspecified.")

    if selection2 is None:
        print("<COOR DIST> Copying selection1 into selection2 because none was provided.")
        selection2 = copy(selection1)

    numexcl = [0, 0, 0] # 0: general, 1: 1-4, 2: nonbond list

    n_atoms_total = get_natom()
    if n_atoms_total == 0:
        return None

    sel1_np = np.array(selection1, dtype=bool)
    sel2_np = np.array(selection2, dtype=bool)

    if sel1_np.shape != (n_atoms_total,) or sel2_np.shape != (n_atoms_total,):
        raise ValueError("Selection arrays do not match the total number of atoms in the system.")

    iatom_indices_orig = np.where(sel1_np)[0]
    jatom_indices_orig = np.where(sel2_np)[0]

    insel = iatom_indices_orig.size
    jnsel = jatom_indices_orig.size

    if insel == 0 or jnsel == 0:
        print(f"<COOR DIST> Selection with no atoms found. selection1 has {insel} atoms, selection2 has {jnsel} atoms.")
        return None
    else:
        print(f"<COOR DIST> {insel} atoms in selection1")
        print(f"<COOR DIST> {jnsel} atoms in selection2")

    all_coords_df = get_positions()
    if all_coords_df.empty:
        return None

    all_coords_np = all_coords_df.to_numpy()

    coords_sel1_np = all_coords_np[iatom_indices_orig, :]
    coords_sel2_np = all_coords_np[jatom_indices_orig, :]

    if resi:
        # Residue-to-residue minimum distance:
        # for each residue pair, find the closest allowed atom pair and report it.

        s1_atom_ids_orig = iatom_indices_orig
        s1_res_ids_str = np.array(atom_info.get_res_ids(s1_atom_ids_orig), dtype=str)
        s1_seg_ids_str = np.array(atom_info.get_seg_ids(s1_atom_ids_orig), dtype=str)
        s1_res_names_str = np.array(atom_info.get_res_names(s1_atom_ids_orig), dtype=str)
        s1_atom_types_str = np.array(atom_info.get_atom_types(s1_atom_ids_orig), dtype=str)

        s2_atom_ids_orig = jatom_indices_orig
        s2_res_ids_str = np.array(atom_info.get_res_ids(s2_atom_ids_orig), dtype=str)
        s2_seg_ids_str = np.array(atom_info.get_seg_ids(s2_atom_ids_orig), dtype=str)
        s2_res_names_str = np.array(atom_info.get_res_names(s2_atom_ids_orig), dtype=str)
        s2_atom_types_str = np.array(atom_info.get_atom_types(s2_atom_ids_orig), dtype=str)

        # Group atoms by (segid, resid) within each selection (store positions into s*_ arrays)
        res1_atoms = {}
        for pos, (seg, resid) in enumerate(zip(s1_seg_ids_str, s1_res_ids_str)):
            key = (seg.strip(), resid.strip())
            res1_atoms.setdefault(key, []).append(pos)

        res2_atoms = {}
        for pos, (seg, resid) in enumerate(zip(s2_seg_ids_str, s2_res_ids_str)):
            key = (seg.strip(), resid.strip())
            res2_atoms.setdefault(key, []).append(pos)

        cutoff_sq = cutoff**2
        icount = 0

        for pos_list1 in res1_atoms.values():
            for pos_list2 in res2_atoms.values():
                min_dist_sq = np.inf
                min_pos_i = None
                min_pos_j = None

                for pos_i in pos_list1:
                    orig_iatom_idx = s1_atom_ids_orig[pos_i]
                    coord_i = all_coords_np[orig_iatom_idx]

                    for pos_j in pos_list2:
                        orig_jatom_idx = s2_atom_ids_orig[pos_j]
                        if orig_iatom_idx == orig_jatom_idx:
                            continue

                        excl_type = qexclt(orig_iatom_idx, orig_jatom_idx)
                        is_general_excl_pair = excl_type == 1
                        is_14_excl_pair = excl_type == 2
                        is_nonbond_excl_pair = False

                        effectively_skipped = False
                        if omit_excl and is_general_excl_pair:
                            numexcl[0] += 1
                            effectively_skipped = True

                        if omit_14excl and is_14_excl_pair:
                            if not (omit_excl and is_general_excl_pair):
                                numexcl[1] += 1
                            effectively_skipped = True

                        if omit_nonbonds and is_nonbond_excl_pair:
                            if not (omit_excl and is_general_excl_pair):
                                numexcl[2] += 1
                            effectively_skipped = True

                        if effectively_skipped:
                            continue

                        diff = coord_i - all_coords_np[orig_jatom_idx]
                        if np.any(np.abs(diff) > cutoff):
                            continue

                        dist_sq = np.dot(diff, diff)
                        if dist_sq <= cutoff_sq and dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            min_pos_i = pos_i
                            min_pos_j = pos_j

                if min_pos_i is None:
                    continue

                dist_val = np.sqrt(min_dist_sq)
                iminAtomId = s1_atom_ids_orig[min_pos_i]
                iminSegId = s1_seg_ids_str[min_pos_i].strip()
                iminResn = s1_res_names_str[min_pos_i].strip()
                iminResId = s1_res_ids_str[min_pos_i].strip()
                iminAtype = s1_atom_types_str[min_pos_i].strip()

                jminAtomId = s2_atom_ids_orig[min_pos_j]
                jminSegId = s2_seg_ids_str[min_pos_j].strip()
                jminResn = s2_res_names_str[min_pos_j].strip()
                jminResId = s2_res_ids_str[min_pos_j].strip()
                jminAtype = s2_atom_types_str[min_pos_j].strip()

                contactDict[icount] = [iminAtomId, iminSegId, iminResn, iminResId, iminAtype,
                                       jminAtomId, jminSegId, jminResn, jminResId, jminAtype,
                                       dist_val]
                icount += 1

        print(f"<COOR DIST> Total Exclusion count ={numexcl[0]:10d}")
        print(f"<COOR DIST> Total 1-4 exclusions  ={numexcl[1]:10d}")
        print(f"<COOR DIST> Total non-exclusions  ={numexcl[2]:10d}")

    else: # if not resi (atom-to-atom distances)
        # Pre-fetch all atom details for selected atoms ONCE
        s1_atom_ids_orig = iatom_indices_orig # These are the original 0-based indices
        s1_res_ids_str = np.array(atom_info.get_res_ids(s1_atom_ids_orig), dtype=str)
        s1_seg_ids_str = np.array(atom_info.get_seg_ids(s1_atom_ids_orig), dtype=str)
        s1_res_names_str = np.array(atom_info.get_res_names(s1_atom_ids_orig), dtype=str)
        s1_atom_types_str = np.array(atom_info.get_atom_types(s1_atom_ids_orig), dtype=str)

        s2_atom_ids_orig = jatom_indices_orig
        s2_res_ids_str = np.array(atom_info.get_res_ids(s2_atom_ids_orig), dtype=str)
        s2_seg_ids_str = np.array(atom_info.get_seg_ids(s2_atom_ids_orig), dtype=str)
        s2_res_names_str = np.array(atom_info.get_res_names(s2_atom_ids_orig), dtype=str)
        s2_atom_types_str = np.array(atom_info.get_atom_types(s2_atom_ids_orig), dtype=str)

        diff_vectors = coords_sel1_np[:, np.newaxis, :] - coords_sel2_np[np.newaxis, :, :]
        too_far_component_wise = np.any(np.abs(diff_vectors) > cutoff, axis=2)
        dist_sq_all = np.sum(diff_vectors**2, axis=2)
        dist_sq_all[too_far_component_wise] = np.inf
        contact_mask = dist_sq_all <= (cutoff**2)

        sel1_contact_indices, sel2_contact_indices = np.where(contact_mask)

        icount = 0
        for i_idx_sel, j_idx_sel in zip(sel1_contact_indices, sel2_contact_indices):
            orig_iatom_idx = iatom_indices_orig[i_idx_sel]
            orig_jatom_idx = jatom_indices_orig[j_idx_sel]

            if orig_iatom_idx == orig_jatom_idx:
                continue

            # Exclusion checks using the coor.qexclt function
            # qexclt expects 0-indexed atom indices
            excl_type = qexclt(orig_iatom_idx, orig_jatom_idx)

            is_general_excl_pair = excl_type == 1
            is_14_excl_pair = excl_type == 2
            is_nonbond_excl_pair = False

            effectively_skipped = False
            # Increment counters if the rule is active AND the pair matches that rule's type
            if omit_excl and is_general_excl_pair:
                 numexcl[0] += 1
                 effectively_skipped = True # If general exclusion is on, it skips

            # 1-4 and nonbond exclusions are more specific.
            # If general exclusion (omit_excl) is active and catches a pair, it's skipped.
            # The other flags (omit_14excl, omit_nonbonds) provide additional, potentially overlapping, criteria.
            # A common interpretation: skip if ANY active omit rule matches.
            # The counts should reflect how many times a pair matched an *active* rule.

            # Re-evaluating skip logic based on CHARMM: a pair is listed if it passes ALL active filters.
            # So, if omit_excl is ON and it's an excluded pair, it's skipped.
            # If omit_14excl is ON and it's a 1-4 pair, it's skipped.
            # If omit_nonbonds is ON and it's a nonbond-excluded pair, it's skipped.

            # Reset effectively_skipped for clearer logic here
            effectively_skipped = False

            # Check for skipping and update counts for *active* rules
            if omit_excl and is_general_excl_pair:
                # numexcl[0] already incremented if this condition is met
                effectively_skipped = True

            if omit_14excl and is_14_excl_pair:
                if not (omit_excl and is_general_excl_pair): # Avoid double counting if already counted by general
                    numexcl[1] +=1
                effectively_skipped = True

            if omit_nonbonds and is_nonbond_excl_pair:
                if not (omit_excl and is_general_excl_pair): # Avoid double counting
                     numexcl[2] +=1
                effectively_skipped = True

            if effectively_skipped:
                continue

            dist_val = np.sqrt(dist_sq_all[i_idx_sel, j_idx_sel])

            iminAtomId = s1_atom_ids_orig[i_idx_sel]
            iminSegId = s1_seg_ids_str[i_idx_sel].strip()
            iminResn = s1_res_names_str[i_idx_sel].strip()
            iminResId = s1_res_ids_str[i_idx_sel].strip()
            iminAtype = s1_atom_types_str[i_idx_sel].strip()

            jminAtomId = s2_atom_ids_orig[j_idx_sel]
            jminSegId = s2_seg_ids_str[j_idx_sel].strip()
            jminResn = s2_res_names_str[j_idx_sel].strip()
            jminResId = s2_res_ids_str[j_idx_sel].strip()
            jminAtype = s2_atom_types_str[j_idx_sel].strip()

            contactDict[icount] = [iminAtomId, iminSegId, iminResn, iminResId, iminAtype,
                                   jminAtomId, jminSegId, jminResn, jminResId, jminAtype,
                                   dist_val]
            icount += 1

        # Print exclusion counts
        print(f"<COOR DIST> Total Exclusion count ={numexcl[0]:10d}")
        print(f"<COOR DIST> Total 1-4 exclusions  ={numexcl[1]:10d}")
        print(f"<COOR DIST> Total non-exclusions  ={numexcl[2]:10d}")


    if not contactDict:
        return None

    contacts_df = pandas.DataFrame.from_dict(contactDict, orient='index',
                                          columns=['atomid_1','segid_1','resn_1','resid_1','type_1',
                                                   'atomid_2', 'segid_2', 'resn_2','resid_2','type_2', 'distance'])
    return contacts_df


class Coordinates:
    __slots__ = 'which_set', 'coords', 'are_dirty'

    def __init__(self, coord_set):
        self.which_set = coord_set
        self.coords = pandas.DataFrame(columns=['x', 'y', 'z', 'w'])
        self.are_dirty = True
        self.pull()

    def pull(self):
        if self.which_set == 'main':
            self.coords = get_main()
        elif self.which_set == 'comp':
            self.coords = get_comparison()
        elif self.which_set == 'comp2':
            self.coords = get_comp2()
        else:
            msg = '{} is not a valid coordinate set'.format(self.which_set)
            raise ValueError(msg)

        self.are_dirty = False
        return self

    def push(self):
        if self.which_set == 'main':
            set_main(self.coords)
        elif self.which_set == 'comp':
            set_comparison(self.coords)
        elif self.which_set == 'comp2':
            set_comp2(self.coords)
        else:
            msg = '{} is not a valid coordinate set'.format(self.which_set)
            raise ValueError(msg)

        self.are_dirty = False
        return self

    def __len__(self):
        return len(self.coords)
