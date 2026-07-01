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
Consists of functions that fetch the residue, segment, chem type, etc. for atoms in a selection.  

"""

import pandas
import pycharmm
import pycharmm.coor as coor
import pycharmm.param as param
import pycharmm.psf as psf


# TODO:
#  1. improve get_atom_table docstring


def get_chem_types(atom_indexes):
    """Get the chemical types for each atom index.

    Parameters
    ----------
    atom_indexes : list[int]
        List of atom indexes 
    Returns
    -------
    list[str]
        A list of the chem_type (str) for each atom index
    
    """
    natc = param.get_natc()
    atc = param.get_atc()
    iac = psf.get_iac()
    n_atoms = psf.get_natom()
    chem_types = list()
    for i in atom_indexes:
        if i >= n_atoms:
            msg = 'atom index {} >= number of atoms {}'
            raise ValueError(msg.format(i, atom_indexes))

        if iac[i] > natc:
            msg = 'No chem type for atom {}'
            raise ValueError(msg.format(i))

        chem_types.append(atc[iac[i]])

    return chem_types


def atom_to_res():
    """Get the residue index of all the atoms in the system.

    Returns
    -------
    list[int]
        A list of residue indexes of the size of the number of atoms
    """
    n_atoms = psf.get_natom()
    n_res = psf.get_nres()
    ibase = psf.get_ibase()
    res_indexes = [-1] * n_atoms
    for i in range(n_res):
        for j in range(ibase[i], ibase[i + 1]):
            res_indexes[j] = i

    return res_indexes


def get_res_indexes(atom_indexes):
    """Get the residue index for select atoms.

    Parameters
    ----------
    atom_indexes : list[int] 
        A list of atom indexes

    Returns
    -------
    list[int]
        List of residue indexes, one for each of `atom_indexes`
    """
    res_indexes = list()
    res_table = atom_to_res()
    for i in atom_indexes:
        try:
            assert i > -1
            assert i < len(res_table)
            res_ind = res_table[i]
        except (IndexError, AssertionError):
            msg = 'No residue available for atom index {}'
            raise ValueError(msg.format(i))

        res_indexes.append(res_ind)

    return res_indexes


def get_res_names(atom_indexes):
    """Get the residue name for each atom index.

    Parameters
    ----------
    atom_indexes : list[int]
        A list of atom indexes

    Returns
    -------
    list[str]
        A list of residue names (str), one for each atom index
    """
    res_indexes = get_res_indexes(atom_indexes)
    res = psf.get_res()
    return [res[i] for i in res_indexes]


def get_res_ids(atom_indexes):
    """Get the residue id for each atom index.

    Parameters
    ----------
    atom_indexes : list[int]
        A list of atom indexes
    Returns
    -------
    list[str]
        A list of residue ids (str), one for each atom index
    """
    res_indexes = get_res_indexes(atom_indexes)
    rid = psf.get_resid()
    return [rid[i] for i in res_indexes]


def atom_to_seg():
    """
    Map atom indexes to seg indexes; used by `get_seg_indexes()`.

    Returns
    -------
    list[int]
        The segment index for all atoms 
    """
    natom = psf.get_natom()
    nseg = psf.get_nseg()
    ibase = psf.get_ibase()
    nictot = psf.get_nictot()
    seg_indexes = [-1] * natom
    for i in range(nseg):
        for j in range(nictot[i], nictot[i + 1]):
            for k in range(ibase[j], ibase[j + 1]):
                seg_indexes[k] = i

    return seg_indexes


def get_seg_indexes(atom_indexes):
    """Get the segment index for each atom index.

    Parameters
    ----------
    atom_indexes : list[int]
        A list of atom indexes

    Returns
    -------
    list[int]
        A list of segment indexes (int), one for each atom index
    """
    seg_indexes = list()
    seg_table = atom_to_seg()
    for k in atom_indexes:
        try:
            assert k > -1
            assert k < len(seg_table)
            seg_ind = seg_table[k]
        except (IndexError, AssertionError):
            msg = 'No segment available for atom index {}'
            raise ValueError(msg.format(k))

        seg_indexes.append(seg_ind)

    return seg_indexes


def get_seg_ids(atom_indexes):
    """Get the segment id (string) for each atom index.

    Parameters
    ----------
    atom_indexes : list[int] 
        A list of atom indexes

    Returns
    -------
    list[str]
        A list of segment ids, one for each atom index
    """
    seg_indexes = get_seg_indexes(atom_indexes)
    sid = psf.get_segid()
    return [sid[i] for i in seg_indexes]


def get_atom_types(atom_indexes):
    """Get the atom name for each atom index.

    Parameters
    ----------
    atom_indexes : list[int] 
        a list of atom indexes

    Returns
    -------
    list[str]
        a list of atom types (str), one for each atom index
    """
    n = psf.get_natom()
    atype = psf.get_atype()
    atom_types = list()
    for i in atom_indexes:
        if i >= n:
            msg = 'atom index {} >= number of atoms {}'
            raise ValueError(msg.format(i, atom_indexes))

        atom_types.append(atype[i])

    return atom_types


def _search(needle, haystack):
    """Find the index of the greatest lower bound for needle in haystack.

    Parameters
    ----------
    needle : int
        you want to find the index of the greatest lower bound for this thing
    haystack : a sorted sequence
        searched elt by elt, for the greatest lower bound for needle

    Returns
    -------
    int
        the first index i in haystack where needle is between elt i and elt i+1
    """
    found = -1
    for i in range(0, len(haystack) - 1):
        if haystack[i] < needle <= haystack[i + 1]:
            found = i
            break

    return found


class AtomInfo:
    """
    Collect all the information stored in CHARMM about an atom index.
    """
    def __init__(self, atom_index):
        """Retrieve all info for atom atom_index

        Parameters 
        ----------
        atom_index : int 
            you're interested in the info for the atom with this index
        """
        # TODO: error checking on atom_index
        self.atom_index = atom_index
        self.atom_type = ''
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

        self.residue_number = -1
        self.residue_name = ''
        self.residue_id = ''

        self.segment_number = -1
        self.segment_id = ''

        self.update()

    def update(self):
        """
        Refill the info for this atom from CHARMM.

        Returns
        -------
        bool    
            True if successful
        """
        # todo: error checking on residue and segment numbers returned
        atypes = psf.get_atype()
        self.atom_type = str(atypes[self.atom_index - 1]).strip()

        positions = coor.get_positions()
        self.x = positions.iloc[self.atom_index - 1]['x']
        self.y = positions.iloc[self.atom_index - 1]['y']
        self.z = positions.iloc[self.atom_index - 1]['z']

        weights = coor.get_weights()
        self.w = weights[self.atom_index - 1]

        self.residue_number = self._get_res()

        res = psf.get_res()
        self.residue_name = str(res[self.residue_number - 1]).strip()

        resid = psf.get_resid()
        self.residue_id = str(resid[self.residue_number - 1]).strip()

        self.segment_number = self._get_seg()

        segid = psf.get_segid()
        self.segment_id = str(segid[self.segment_number - 1]).strip()

        return True

    def _get_res(self):
        """
        Get residue index for atom index.

        Returns
        -------
        int
            residue index
        """
        ibase = psf.get_ibase()

        # TODO: put in try
        i = _search(self.atom_index, ibase)
        return i + 1

    def _get_seg(self):
        """Get segment index for atom index.

        Returns
        -------
        int
            segment index 
        """
        nictot = psf.get_nictot()

        # TODO: put in try
        i = _search(self.residue_number, nictot)
        return i + 1


def get_atom_table(selection=None):
    """Get a table of (atom index) X (chem type, res name, seg id) for atoms in a selection.

    Parameters
    ----------
    selection : pycharmm.SelectAtoms
        fill table for only these atoms, all atoms if None

    Returns
    -------
    pandas.DataFrame
        a data frame with info for each atom
    """
    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    atoms = selection.get_atom_indexes()
    atom_table = pandas.DataFrame({
        'index': [a.atom_index for a in atoms],
        'type': [a.atom_type for a in atoms],
        'residue_number': [a.residue_number for a in atoms],
        'residue_name': [a.residue_name for a in atoms],
        'residue_id': [a.residue_id for a in atoms],
        'segment_number': [a.segment_number for a in atoms],
        'segment_id': [a.segment_id for a in atoms],
        'x': [a.x for a in atoms],
        'y': [a.y for a in atoms],
        'z': [a.z for a in atoms],
        'w': [a.w for a in atoms]})

    return atom_table

