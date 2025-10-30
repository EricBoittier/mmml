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

"""Select and manipulate sets of atoms.

Corresponds to CHARMM command `SELEct`
See [SELEct documentation](https://academiccharmm.org/documentation/version/c47b1/select)

Examples
========
Import module
>>> import pycharmm
>>> import pycharmm.selection as sel
>>> import pycharmm.lingo as lingo

Select all the hydrogen atoms and store it in a selection called `HYD`
>>> sel.store_selection('HYD', sel.hydrogen())

Check to see if there is a selection saved by the name `HYD`
>>> sel.find('HYD')

Obtain `COOR STAT` on the selection of atoms called `HYD`
>>> lingo.charmm_script('coor stat sele HYD end')

"""


import ctypes
import math
import typing
from collections.abc import Iterable

import numpy as np

import pycharmm
import pycharmm.coor as coor
import pycharmm.lib as lib
import pycharmm.param as param
import pycharmm.psf as psf

import pycharmm.atom_info as atom_info


Selection = typing.Tuple[bool]


def or_selection(sel_a: Selection, sel_b: Selection) -> Selection:
    """Use eltwise logical `or` to produce a new selection.
    """
    return tuple(elt_a or elt_b for elt_a, elt_b in zip(sel_a, sel_b))


def and_selection(sel_a: Selection, sel_b: Selection) -> Selection:
    """Use eltwise logical `and` to produce a new selection
    """
    return tuple(elt_a and elt_b for elt_a, elt_b in zip(sel_a, sel_b))


def not_selection(sel: Selection) -> Selection:
    """Use eltwise logical `not` to produce a new selection
    """
    return tuple(not elt for elt in sel)


def none_selection(size: int) -> Selection:
    """Get a new selection in which all elements are `False`
    """
    return (False, ) * size


def all_selection(size: int) -> Selection:
    """get a new selection in which all elements are `True`
    """
    return (True, ) * size


def by_atom_inds(inds: Iterable, selection: Selection) -> Selection:
    """Copy selection to new_sel then set new_sel[`inds`] to `True`
    """
    new_sel = tuple(True if i in inds else val
                    for i, val in enumerate(selection))
    return new_sel


def by_residue_name(residue_name: str) -> Selection:
    """Select all atoms in a residue

    Parameters
    ----------
    residue_name : string
                   name of residue to select

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    res = psf.get_res()
    ibase = psf.get_ibase()
    select_inds = tuple()
    for i, name in enumerate(res):
        if name == residue_name:
            for j in range(ibase[i], ibase[i + 1]):
                select_inds = select_inds + (j, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def by_residue_id(residue_id: str) -> Selection:
    """Select all atoms in a residue by residue id

    Parameters
    ----------
    residue_id : str
                 CHARMM id of residue to select

    Returns
    -------
    flags : Selection
            atom i selected <==> flags[i] == True
    """
    ids = psf.get_resid()
    ibase = psf.get_ibase()
    select_inds = tuple()
    for i, name in enumerate(ids):
        if name == residue_id:
            for j in range(ibase[i], ibase[i + 1]):
                select_inds = select_inds + (j, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def by_segment_id(segment_id: str) -> Selection:
    """Select all atoms in a segment.

    Parameters
    ----------
    segment_id : str
                 name of segment to select

    Returns
    -------
    flags : Selection
            atom i selected <==> flags[i] == True
    """
    segids = psf.get_segid()
    ibase = psf.get_ibase()
    nictot = psf.get_nictot()
    select_inds = tuple()
    for i, name in enumerate(segids):
        if name == segment_id:
            for j in range(nictot[i], nictot[i + 1]):
                for k in range(ibase[j], ibase[j + 1]):
                    select_inds = select_inds + (k, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def by_atom_type(atom_type: str) -> Selection:
    """Select all atoms of type `atom_type`

    Parameters
    ----------
    atom_type : string
                IUPAC name of type to select

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    atom_types = psf.get_atype()
    select_inds = tuple()
    for i, current_type in enumerate(atom_types):
        if atom_type == current_type:
            select_inds = select_inds + (i, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def by_chem_type(chem_type: str) -> Selection:
    """Select all atoms of param type code `chem_type`

    Parameters
    ----------
    chem_type : str
                parameter type code to select

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    natc = param.get_natc()
    atc = param.get_atc()
    iac = psf.get_iac()
    select_inds = tuple()
    n_atoms = psf.get_natom()
    for i in range(n_atoms):
        if iac[i] > natc:
            raise ValueError('No VDW parameters available for CHEM token '
                             + chem_type)

        if chem_type == atc[iac[i]]:
            select_inds = select_inds + (i, )

    return by_atom_inds(select_inds, none_selection(n_atoms))


def all_atoms() -> Selection:
    """Select all atoms.

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    return all_selection(n_atoms)


def no_atoms() -> Selection:
    """Select no atoms.

    Returns
    -------
    flags : boolean tuple
            flags[i] = False for all atoms
    """
    n_atoms = psf.get_natom()
    return none_selection(n_atoms)


def by_residue_atom(segid: str, resid: str, atype: str) -> Selection:
    """Select all atoms in single residue with IUPAC name `atype` in residue with ID `resid` in a segment with ID `segid`.

    Parameters
    ----------
    segid : str
            segment identifier (A1, MAIN, ...)
    resid : str
            residue identifier (1, 23, 45B, ...)
    atype : str
            an IUPAC name

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    segids = psf.get_segid()
    resids = psf.get_resid()
    atypes = psf.get_atype()
    ibase = psf.get_ibase()
    nictot = psf.get_nictot()
    select_inds = tuple()
    for seg_i, name in enumerate(segids):
        if segid == name:
            for res_i in range(nictot[seg_i], nictot[seg_i + 1]):
                if resid == resids[res_i]:
                    for atom_i in range(ibase[res_i], ibase[res_i + 1]):
                        if atype == atypes[atom_i]:
                            select_inds = select_inds + (atom_i, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def by_point(x: float, y: float, z: float,
             cut=8.0, periodic=False) -> Selection:
    """Selects all atoms within a sphere around point (`x`,`y`,`z`) with radius `cut`

    Parameters
    ----------
    x : float
        x coord of selection sphere center
    y : float
        y coord of selection sphere center
    z : float
        z coord of selection sphere center
    cut : float, default = 8.0
          radius of selection sphere
    periodic : bool, default = False
               if simple periodic boundary conditions are in effect
               through the use of the MIPB command,
               the selection reflects the appropriate periodic boundaries

    Returns
    -------
    flags : boolean tuple
            atom i selected <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    select_inds = tuple()
    positions = coor.get_positions()
    for i in range(n_atoms):
        pos_i = positions.iloc[i]
        dx = x - pos_i['x']
        dy = y - pos_i['y']
        dz = z - pos_i['z']

        if periodic:
            is_to_box = lib.charmm.pbound_is_to_box()
            is_cubic_box = lib.charmm.pbound_is_cubic_box()
            if is_to_box.value == 1 or is_cubic_box.value == 1:
                boxinv_x = ctypes.c_double(0.0)
                boxinv_y = ctypes.c_double(0.0)
                boxinv_z = ctypes.c_double(0.0)
                lib.charmm.pbound_get_boxinv(ctypes.byref(boxinv_x),
                                             ctypes.byref(boxinv_y),
                                             ctypes.byref(boxinv_z))
                dx *= boxinv_x
                if dx > 0.5:
                    dx -= 1.0

                if dx < -0.5:
                    dx += 1.0

                dy *= boxinv_y
                if dy > 0.5:
                    dy -= 1.0

                if dy < -0.5:
                    dy += 1.0

                dz *= boxinv_z
                if dz > 0.5:
                    dz -= 1.0

                if dz < -0.5:
                    dz += 1.0

                if is_to_box.value == 1:
                    lib.charmm.pbound_get_r75.restype = ctypes.c_double
                    r75 = lib.charmm.pbound_get_r75()
                    corr = 0.5 * math.trunc(r75 *
                                            (abs(dx) + abs(dy) + abs(dz)))
                    dx -= math.copysign(corr, dx)
                    dy -= math.copysign(corr, dy)
                    dz -= math.copysign(corr, dz)

                size_x = ctypes.c_double(0.0)
                size_y = ctypes.c_double(0.0)
                size_z = ctypes.c_double(0.0)
                lib.charmm.pbound_get_size(ctypes.byref(size_x),
                                           ctypes.byref(size_y),
                                           ctypes.byref(size_z))
                dx *= size_x
                dy *= size_y
                dz *= size_z
            else:
                dx = ctypes.c_double(dx)
                dy = ctypes.c_double(dy)
                dz = ctypes.c_double(dz)
                lib.charmm.pbound_pbmove(ctypes.byref(dx),
                                         ctypes.byref(dy),
                                         ctypes.byref(dz))

        dist_sq = np.sqrt (dx * dx + dy * dy + dz * dz )
        if dist_sq <= cut:
            select_inds = select_inds + (i, )

    return by_atom_inds(select_inds, none_selection(n_atoms))


def is_hydrogen(i: int) -> bool:
    """True if atom `i` is hydrogen

    Parameters
    ----------
    i : integer
        index of atom to test

    Returns
    -------
    answer : bool
       atom i is hydrogen <==> answer == True
    """
    c_i = ctypes.c_int(i + 1)
    test = lib.charmm.select_is_hydrog(ctypes.byref(c_i))
    answer = False
    if test == 1:
        answer = True

    return answer


def is_lone(i: int) -> bool:
    """True if atom `i` is a lonepair

    Parameters
    ----------
    i : integer
        index of atom to test

    Returns
    -------
    answer : bool
            atom i is a lonepair <==> answer == True
    """
    c_i = ctypes.c_int(i + 1)
    test = lib.charmm.select_is_lone(ctypes.byref(c_i))
    answer = False
    if test == 1:
        answer = True

    return answer


def is_initial(i: int) -> bool:
    """True if atom `i` has known coords

    Parameters
    ----------
    i : integer
        index of atom to test

    Returns
    -------
    answer : bool
            atom `i` has known coords <==> answer == True
    """
    c_i = ctypes.c_int(i + 1)
    test = lib.charmm.select_is_initial(ctypes.byref(c_i))
    answer = False
    if test == 1:
        answer = True

    return answer


def hydrogen() -> Selection:
    """Selects all hydrogen atoms

    Returns
    -------
    flags : boolean tuple
            atom `i` hydrogen <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    select_inds = tuple(i for i in range(n_atoms) if is_hydrogen(i))
    return by_atom_inds(select_inds, none_selection(n_atoms))


def initial() -> Selection:
    """Selects all atoms with known coords

    Returns
    -------
    flags : boolean tuple
            atom `i` has known coords <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    select_inds = tuple(i for i in range(n_atoms) if is_initial(i))
    return by_atom_inds(select_inds, none_selection(n_atoms))


def lone() -> Selection:
    """Selects all lonepair atoms

    Returns
    -------
    flags : boolean tuple
            atom `i` lonepair <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    select_inds = tuple(i for i in range(n_atoms) if is_lone(i))
    return by_atom_inds(select_inds, none_selection(n_atoms))


def get_property(prop_name):
    """Return array filled with `natoms` numeric property *prop*

    Parameters
    ----------
    prop_name : str
                identifier for numeric property of atoms

    Returns
    -------
    prop_vals : numeric list
                numeric value for each atom i representing *prop*
    """
    c_prop = ctypes.c_char_p(prop_name.encode('utf-8').lower())
    n_atoms = psf.get_natom()
    prop_vals = (ctypes.c_double * n_atoms)(0.0)
    lib.charmm.select_get_property(c_prop, prop_vals,
                                   ctypes.c_int(n_atoms))
    prop_vals = list(prop_vals)
    return prop_vals


def prop(prop_name, func: typing.Callable[[float, float], bool], tol) -> Selection:
    """Select all atoms for which `func(tol, prop_val)` is True

    Parameters
    ----------
    prop_name : str
                identifier for numeric property of atoms
    func : function
           of two arguments, `func(tol, prop_val)`
    tol : float
          tolerance to pass to func as first argument

    Returns
    -------
    flags : boolean tuple
            atom `i` selected <==> flags[i] == True
    """
    select_inds = tuple(i for i, p in enumerate(get_property(prop_name))
                        if func(tol, p))
    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))


def residues(resname_a: str, resname_b='') -> Selection:
    """Select all atoms in a range of residues

    Parameters
    ----------
    resname_a : str
                name of first residue in range
    resname_b : str, default = ''
                name of last residue in range

    Returns
    -------
    flags : boolean tuple
            flags[i] == True means that atom `i` is selected
    """
    c_name_a = ctypes.c_char_p(resname_a.encode('utf-8'))
    if resname_b:
        c_name_b = ctypes.c_char_p(resname_b.encode('utf-8'))
    else:
        c_name_b = ctypes.c_char_p(resname_a.encode('utf-8'))

    natom = psf.get_natom()
    flags = (ctypes.c_int * natom)(0 * natom)
    lib.charmm.select_resname_range(c_name_a, c_name_b, flags)
    return tuple(True if flag == 1 else False for flag in flags)


def segments(segid_a: str, segid_b='') -> Selection:
    """Select all atoms in a range of segments

    Parameters
    ----------
    segid_a : str
              name of first segment in range
    segid_b : str, default = ''
              name of last segment in range

    Returns
    -------
    flags : boolean tuple
            flags[i] == True means that atom `i` is selected
    """
    c_name_a = ctypes.c_char_p(segid_a.encode('utf-8'))
    if segid_b:
        c_name_b = ctypes.c_char_p(segid_b.encode('utf-8'))
    else:
        c_name_b = ctypes.c_char_p(segid_a.encode('utf-8'))

    natom = psf.get_natom()
    flags = (ctypes.c_int * natom)(0 * natom)
    lib.charmm.select_segid_range(c_name_a, c_name_b, flags)
    return tuple(True if flag == 1 else False for flag in flags)


def whole_residues(sel: Selection) -> Selection:
    """select the whole residue of each atom in a selection

    Parameters
    ----------
    sel : Selection
        selection of atoms, each atom's whole residue will be selected
    Returns
    -------
    boolean tuple
        a new selection of residues
    """
    residue_table = atom_info.atom_to_res()
    residues_wanted = atom_info.get_res_indexes([i for i, v in enumerate(sel) if v])
    new_sel = none_selection(len(sel))
    new_indexes = [i for i, v in enumerate(residue_table) if v in residues_wanted]
    return by_atom_inds(new_indexes, new_sel)


def around(sel: Selection, r_cut: float) -> Selection:
    """Select all atoms within `r_cut` of the current selection

    This is equivalent to CHARMM's ".around." command.
    Implemented as a linked-list cell algorithm.
    More information at: https://doi.org/10.1017/CBO9780511816581
    and ISBN: 9780122673511

    Parameters
    ----------
    sel : Selection
        atoms around which a selection of atoms is desired
    r_cut : float
        Selection cut-off

    Raise
    -----
        ValueError if `r_cut` is 0 angstroms or less

    Returns
    -------
    boolean tuple
        Essentially a new `SelectAtoms` object is returned. It contains the new selection. **The new selection includes the current selection.**

    Example
    -------
    >>> # select all water molecules that are 2.8 angstroms from the protein
    >>> example_sel = pycharmm.SelectAtoms(segid="TIP3") & pycharmm.SelectAtoms(segid="PROTEIN").around(2.8)
    """
    if r_cut <= 0:
        raise ValueError("r_cut should be greater than 0 angstroms!")

    # get dimensions of entire system (not just selection)
    stats = coor.stat()
    lx = stats["xmax"] - stats["xmin"]
    ly = stats["ymax"] - stats["ymin"]
    lz = stats["zmax"] - stats["zmin"]
    system_length = max(lx, ly, lz)

    rn = system_length / int(system_length / r_cut)  # cell length

    # number of cells along an axis
    # we pad with a two extra cells (+2) for the edge case
    # where atom is very close to boundary
    sc = int(system_length / rn) + 2

    # relative neighborhood array
    d_half = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 1, 0],
                       [0, 1, 0], [0, 0, 1], [-1, 0, 1], [1, 0, 1], [-1, -1, 1],
                       [0, -1, 1], [1, -1, 1], [-1, 1, 1], [0, 1, 1], [1, 1, 1]])
    d = np.unique(np.concatenate((d_half, -d_half)), axis=0)  # get all 27 neighbors including self cell (0,0,0)

    pos = coor.get_positions()
    r = pos.to_numpy()  # 3d array coordinates of our system
    pos_sel = pos[list(sel)]
    r_sel_indices = tuple(pos_sel.index)  # get the selected atom indices
    c = np.floor([[i[0] / rn, i[1] / rn, i[2] / rn] for i in r]).astype(
        np.int_)  # N*3 array of cell indices for all atoms (each atom assigned to a cell)
    natom = psf.get_natom()
    ll = [0] * natom
    head = np.full((sc, sc, sc), -1)
    near = list()  # atoms that are nearby

    # create our linked list and header list
    for i, icell in enumerate(c):
        # icell is index of cell, i is the atom index
        ll[i] = int(head[icell[0]][icell[1]][icell[2]])  # store
        head[icell[0]][icell[1]][icell[2]] = int(i)

    for i in r_sel_indices:  # loop through our selection
        icell = c[i]  # get cell that selected atom is in
        for dj in d:
            jcell = icell + dj
            # jcell = np.mod(jcell, sc)  # apply periodic conditions so that if jcell contains index > sc, it wraps back
            j = int(head[jcell[0]][jcell[1]][jcell[2]])
            while j >= 0:  # while the atom chain still continues
                dr = np.linalg.norm(r[i] - r[j])
                if dr <= r_cut:
                    near.append(j)

                j = ll[j]

    near = list(set(near))  # remove duplicates
    sel_near = [False] * natom  # convert list of atom indices to atom selection boolean list
    for i in near:
        sel_near[i] = True

    return tuple(sel_near)


def get_max_name() -> int:
    """Ask CHARMM for the max len of the name of a stored selection.
    Returns
    -------
    max_name : int
    """
    max_name = lib.charmm.select_get_max_name()
    return max_name


def get_num_stored() -> int:
    """Ask CHARMM for the number of stored selections
    Returns
    -------
    num_stored : int
    """
    num_stored = lib.charmm.select_get_num_stored()
    return num_stored


def find(name: str) -> int:
    """Get the index of the named stored selection
    Returns
    -------
    int
    """
    c_name = ctypes.c_char_p(name.encode('utf-8'))
    c_len_name = ctypes.c_int(len(name))
    found = lib.charmm.select_find(c_name, c_len_name)
    return found


def store_selection(name: str, sel: Selection) -> str:
    """Store selection in CHARMM as `name`
    Parameters
    ----------
    name : str
        Name to be assigned to the stored selection
    sel : boolean tuple
        Selection of atoms
    Returns
    -------
    name : str
        Same as the `name` in Parameters

    Example
    -------
    In CHARMM, one can name a selection in the following way
    >>> DEFIne sel1 select type C end

    The equivalent in pycharmm, using the store_selection function, can be
    accomplished using the following lines

    >>> import pycharmm
    >>> import pycharmm.selection as sel
    >>> sel.store_selection('sel1', sel.by_atom_type('C'))

    """
    c_name = ctypes.c_char_p(name.upper().encode('utf-8'))
    c_len_name = ctypes.c_int(len(name))

    c_sel = (ctypes.c_int * len(sel))(*sel)
    c_len_sel = ctypes.c_int(len(sel))

    lib.charmm.select_store(c_name, c_len_name, c_sel, c_len_sel)
    print('A selection has been stored as {}'.format(name.upper()))
    return name


def get_stored_names() -> typing.List[str]:
    """Get a list of all the names of the selections stored in CHARMM
    Returns
    -------
    string list
    """
    n = get_num_stored()
    max_name = get_max_name()
    name_buffers = [ctypes.create_string_buffer(max_name) for _ in range(n)]
    name_pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof,
                                               name_buffers))

    lib.charmm.select_get_stored_names(name_pointers, ctypes.c_int(n), ctypes.c_int(max_name))
    names = [b.value.decode(errors='ignore') for b in name_buffers[0:n]]
    return names


def delete_stored_selection(name: str) -> str:
    """Remove the named selection from CHARMM
    Returns
    -------
    str
        Name of the stored selection asked for removal
    """
    c_name = ctypes.c_char_p(name.encode('utf-8'))
    c_len_name = ctypes.c_int(len(name))
    lib.charmm.select_delete(c_name, c_len_name)
    return name


# Addition Kai Toepfer May 2022
def by_atom_num(num: int) -> Selection:
    """Select atoms of index number

    Parameters
    ----------
    num : int
        atom number

    Returns
    -------
    flags : boolean tuple
            atom `i` selected <==> flags[i] == True
    """
    select_inds = (num, )

    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))

# Addition Arghya Argo Chakravorty June 2023
def bonded(selection: Selection) -> Selection:
    """Select all atoms bonded to atoms in the current selection
       and include the current selection in the output.

       Equivalent to the .BONDED. token in CHARMM's selection.

       Parameters
       ----------
       selection: An object of type `Selection` i.e. `typing.Tuple[bool]`

       Returns
       -------
       An object of type `Selection`
    """

    nbond = psf.get_nbond()
    ib, jb = psf.get_ib_jb()
    """
    print(ib)
    print('------')
    print(jb)
    print('------')
    """
    bondedAtomSet = set()

    for ibond in range(nbond):

        # what if the atom is the 1st atom in the bond
        iat2 = ib[ibond]
        qindx2 = iat2 - 1
        if selection[qindx2]:
            # add the atom and its partner in the bond
            bondedAtomSet.add(qindx2)
            bondedAtomSet.add(jb[ibond] - 1)

        # what if the atom is the 2nd atom in the bond?
        iat2 = jb[ibond]
        qindx2 = iat2 - 1
        if selection[qindx2]:
            # add the atom and its partner in the bond
            bondedAtomSet.add(qindx2)
            bondedAtomSet.add(ib[ibond] - 1)

    select_inds = tuple()
    for val in bondedAtomSet:
        select_inds = select_inds + (val,)


    n_atoms = psf.get_natom()
    return by_atom_inds(select_inds, none_selection(n_atoms))
