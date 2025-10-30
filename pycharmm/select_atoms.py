"""Functions for selecting atoms in a structure 

Corresponds to CHARMM command `SELEction`

See CHARMM documentation [select](<https://academiccharmm.org/documentation/version/c47b1/select>)
for more information

Examples
========
>>> import pycharmm

Select all atoms in a protein, whose segment ID is PROT
>>> sele_prot = pycharmm.SelectAtoms(seg_id='PROT')

Select all CA atoms
>>> sele_ca = pycharmm.SelectAtoms(atom_type='CA')

Select all CA atoms of the protein
>>> sele_prot_ca = sele_prot & sele_ca

Save the above selection in CHARMM with name prot_ca. 
This is similar to CHARMM command:

`define prot_ca sele segid PROT .and. type CA end`
>>> sele_prot_ca.store('prot_ca')

For segments ALAD and GLAD, select all atoms whose residue IDs 
are either 1 or 2, and whose atom names are either N or CA
>>> atoms = pycharmm.SelectAtoms().by_res_and_type('ALAD GLAD','1 2','N CA') 


"""

import ctypes
import random
import string
import typing

import numpy

import pycharmm.psf as psf
import pycharmm.select as select
import pycharmm.atom_info as atom_info

def _get_random_string(str_size):
    return ''.join(random.choice(string.ascii_letters)
                   for _ in range(str_size)).upper()


def _generate_script_name():
    max_name = select.get_max_name()
    rand_name = _get_random_string(max_name)
    found = select.find(rand_name)
    iter_limit = 100
    while found > 0 and iter_limit > 0:
        rand_name = _get_random_string(max_name)
        found = select.find(rand_name)
        iter_limit -= 1

    if iter_limit <= 0 < found:
        raise RuntimeError('For this selection, ' +
                           'tried 100 random names, and ' +
                           'could not find an unused random name.')

    return rand_name


class SelectAtoms:
    def __init__(self, selection=None,
                 select_all=False,
                 seg_id='',
                 res_id='', res_name='',
                 atom_nums=None,
                 atom_type='', chem_type='',
                 initials=False,
                 lonepairs=False,
                 hydrogens=False,
                 update=True):
        """
        Parameters
        ----------
        selection : bool tuple
            A boolean tuple whose length is the number of atoms
            in the system. True if the corresponding atom is selected.
        select_all : bool
            True for selecting all atoms
        seg_id : str
            segment ID of the selection
        res_id : str
            residue ID of the selection
        res_name : str
            residue name of the selection
        atom_nums : int or list of int
            atom index, i.e., atom number 
        atom_type : str
            atom names of the selection
        chem_type : str
            chemical type of the selection
        initials : bool
            True for selecting all atoms with known coordinates
        lonepairs : bool
            True for selecting all lone pairs
        hydrogens : bool
            True for selecting all hydrogen atoms
        update : bool
            True for updating the selection
            
        """
        self._name = ''
        self._stored = False
        self._do_update = update
        self._selection = None
        n_atoms = psf.get_natom()
        if selection:
            self.set_selection(selection)
        else:
            self.set_selection(select.none_selection(n_atoms))

        if select_all:
            self.set_selection(select.all_selection(n_atoms))

        if seg_id:
            self.by_seg_id(seg_id)

        if res_id:
            self.by_res_id(res_id)

        if res_name:
            self.by_res_name(res_name)
        
        if not atom_nums is None:
            if isinstance(atom_nums, int):
                self.by_atom_nums([atom_nums])
            else:
                self.by_atom_nums(atom_nums)
        
        if atom_type:
            self.by_atom_type(atom_type)

        if chem_type:
            self.by_chem_type(chem_type)

        if initials:
            self.all_initial_atoms()

        if lonepairs:
            self.all_lonepair_atoms()

        if hydrogens:
            self.all_hydrogen_atoms()

    def _select_atom(self, atom_i):
        old_val = self[atom_i]
        self.set_selection(select.by_atom_inds(self.get_selection(), atom_i))
        return old_val

    def _deselect_atom(self, atom_i):
        old_val = self[atom_i]
        new_sel = tuple(sel if not ind == atom_i else False
                        for ind, sel in enumerate(self.get_selection()))
        self.set_selection(new_sel)
        return old_val

    def get_selection(self) -> select.Selection:
        """
        For a pycharmm selection, return a boolean tuple, whose length
        is the number of atoms in the system.

        True if an atom is in the selection, otherwise False.
        """
        return self._selection

    def set_selection(self, selection: select.Selection):
        """
        For a pycharmm selection, update it based on the input boolean tuple.

        Parameters
        ----------
        selection: boolean tuple
            Length of the tuple is equal to the number of atoms in the system

            True if an atom is in the selection, otherwise False.

        """
        self._selection = selection
        if self._do_update:
            self._update()
        else:
            print('WARNING: Atom properties for ' +
                  'this selections are not up to date.')

        return self

    def by_seg_id(self, seg_id):
        """Select by segment ID

        Parameters
        ----------
        seg_id : str
              segment ID
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_segment_id(seg_id))
        self.set_selection(new_sel)
        return self

    def by_res_id(self, res_id):
        """Select by residue ID

        Parameters
        ----------
        res_id : str
              residue ID
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_residue_id(res_id))
        self.set_selection(new_sel)
        return self

    def by_res_name(self, res_name):
        """Select by residue name

        Parameters
        ----------
        res_name : str
              residue name
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_residue_name(res_name))
        self.set_selection(new_sel)
        return self

    def by_chem_type(self, chem_type):
        """Select by chemical type 

        Parameters
        ----------
        chem_type : str
              chemical type 
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_chem_type(chem_type))
        self.set_selection(new_sel)
        return self
    
    def by_atom_nums(self, atom_nums):
        """Select by atom index 

        Parameters
        ----------
        atom_nums : list of int
              atom index, i.e., atom number 

              Note that atom index starts from 0
        """
        for num in atom_nums:
            new_sel = select.or_selection(
                self.get_selection(), select.by_atom_num(num))
            self.set_selection(new_sel)
        return self

    def by_atom_type(self, atom_type):
        """Select by atom name

        Parameters
        ----------
        atom_type : str
              atom name 
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_atom_type(atom_type))
        self.set_selection(new_sel)
        return self

    def in_sphere(self, x, y, z, radius=0.8, is_periodic=False):
        """ Select all atoms within a sphere around point (x,y,z) with a certain radius.

        Corresponds to CHARMM command `sele point`

        Parameters
        ----------
        x : float
              x coordinate of the reference point
        y : float
              y coordinate of the reference point
        z : float
              z coordinate of the reference point
        radius : float 
              radius of the shpere around a given point
        is_periodic : bool
              If True AND simple periodic boundary conditions are in effect
              through the use of the MIPB command, the selection reflects
              the appropriate periodic boundaries.
              see [images](<https://academiccharmm.org/documentation/version/c47b1/images/>)
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.by_point(x, y, z,
                                                      radius,
                                                      is_periodic))
        self.set_selection(new_sel)
        return self

    def by_res_and_type(self, seg_id, res_id, atom_type):
        """Select multiple segments, resids and atom types. Specifically,
        for all segments whose names are in `seg_id`, select all atoms whose
        residue IDs are in `res_id` and whose atom names are in `atom_type`.

        Parameters
        ----------
        seg_id : string
                 string of segment identifiers ('A1 MAIN SEG1 ... SEGn')
        res_id : string
                 string of residue identifiers ('1 3 5 6 ...n')
        atom_type : string
                    string of an IUPAC names ('C CA CB N S')

        Returns
        -------
        flags : boolean list
                atom i selected <==> flags[i] == True
        """
        seg = SelectAtoms()
        for segid in seg_id.strip().split():
            seg.by_seg_id(segid)
        res = SelectAtoms()
        for resid in res_id.strip().split():
            res.by_res_id(res_id=resid)
        atoms = SelectAtoms()
        for atomname in atom_type.strip().split():
            atoms.by_atom_type(atom_type=atomname)
        my_atom = seg & res & atoms
        new_sel = select.or_selection(self.get_selection(),
                                      my_atom.get_selection())
        self.set_selection(new_sel)
        return self

    def all_atoms(self):
        """
        Select all atoms in the system
        """
        self.set_selection(select.all_atoms())
        return self

    def all_initial_atoms(self):
        """
        Select all atoms with known coordinates
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.initial())
        self.set_selection(new_sel)
        return self

    def all_lonepair_atoms(self):
        """
        Select all lonepairs 
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.lone())
        self.set_selection(new_sel)
        return self

    def all_hydrogen_atoms(self):
        """
        Select all hydrogen atoms
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.hydrogen())
        self.set_selection(new_sel)
        return self

    def by_property(self, prop_name: str,
                    func: typing.Callable[[float, float], bool],
                    tol: float):
        """
        Select based on atom properties 
        """
        new_sel = select.or_selection(self.get_selection(),
                                      select.prop(prop_name,
                                                  func,
                                                  tol))
        self.set_selection(new_sel)
        return self

    def around(self, radius):
        """
        Finds all atoms within a `radius` around the atoms specified
        in the selection
        """
        new_sel = select.around(self.get_selection(), radius)
        self.set_selection(new_sel)
        return self

    def whole_residues(self):
        """
        Select by residues as a whole
        """
        new_sel = select.whole_residues(self.get_selection())
        self.set_selection(new_sel)
        return self

    def __list__(self):
        return list(self.get_selection())

    def __array__(self) -> numpy.ndarray:
        return numpy.array(list(self))

    def as_ctypes(self):
        """
        For a pycharmm selection, convert it to a selection 
        for lib.charmm to use
        """
        atoms = [int(atom) for atom in list(self)]
        natoms = len(atoms)
        c_selection = (ctypes.c_int * natoms)(*atoms)
        return c_selection

    def __len__(self):
        return len(self.get_selection())

    def __and__(self, other):
        new_sel = select.and_selection(self.get_selection(),
                                       other.get_selection())
        return SelectAtoms(new_sel)

    def __or__(self, other):
        new_sel = select.or_selection(self.get_selection(),
                                      other.get_selection())
        return SelectAtoms(new_sel)

    def __invert__(self):
        new_sel = select.not_selection(self.get_selection())
        return SelectAtoms(new_sel)

    def __iter__(self):
        return SelectAtomsIterator(self)

    def __getitem__(self, key):
        return self.get_selection()[key]

    def is_selected(self, atom_index) -> bool:
        """
        Check if an atom is in a pycharmm selection

        Parameters
        ----------
        atom_index : int
            atom index

            Note that atom index starts from 0

        Returns
        -------
        is_sel_i : bool
            True if the atom is in the pycharmm selection
        """
        is_sel_i = False
        if atom_index < len(self.get_selection()):
            is_sel_i = self[atom_index]

        return is_sel_i

    def is_stored(self):
        """
        Check if a pycharmm selection has been stored in CHARMM

        Returns
        -------
        self._stored : bool
             True if a pycharmm selection has been stored in CHARMM with
             a name
        """
        return self._stored

    def get_stored_name(self):
        """Get the name of the pycharmm selection in CHARMM
        
        Returns
        -------
        name : str
             name of the pycharmm selection in CHARMM
        """
        name = ''
        if self.is_stored():
            name = self._name

        return name

    def store(self, name=''):
        """
        Save the pycharmm selection to CHARMM 

        Parameters
        ----------
        name : str
            name of pycharmm selection in CHARMM
        """
        max_name = select.get_max_name()
        if name:
            self._name = name[:max_name]

        if not self._name:
            self._name = _generate_script_name()

        if len(self._name) > max_name:
            self._name = self._name[:max_name]

        select.store_selection(self._name, self.get_selection())
        self._stored = True
        return self._name

    def unstore(self):
        """
        Unstore/remove the named pycharmm selection in CHARMM

        Returns
        -------
        was_stored : old store status
        """
        was_stored = self.is_stored()
        if was_stored:
            select.delete_stored_selection(self._name.upper())
            self._stored = False

        return was_stored

    def __enter__(self):
        self.store()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unstore()

    def _update(self):
        sel = self.get_selection()
        inds = [i for i, x in enumerate(sel) if x]
        self._atom_indexes = inds
        self._n_selected = len(inds)
        self._chem_types = atom_info.get_chem_types(inds)
        self._res_indexes = atom_info.get_res_indexes(inds)
        self._res_names = atom_info.get_res_names(inds)
        self._res_ids = atom_info.get_res_ids(inds)
        self._seg_indexes = atom_info.get_seg_indexes(inds)
        self._seg_ids = atom_info.get_seg_ids(inds)
        self._atom_types = atom_info.get_atom_types(inds)
        if self.is_stored():
            self.store()

        return self

    def get_atom_indexes(self):
        """Get a list of atom indexes for selected atoms.

        Note that atom index starts from 0
        """
        return self._atom_indexes[:]

    def get_n_selected(self):
        """ Get number of selected atoms
        """
        return self._n_selected

    def get_chem_types(self):
        """Get a list of chemical types (based on the topology file)
        for selected atoms
        """
        return self._chem_types[:]

    def get_res_indexes(self):
        """Get a list of residue indexes for selected atoms.

        Note that residue index starts from 0
        """
        return self._res_indexes[:]

    def get_res_names(self):
        """Get a list of residue names for selected atoms
        """
        return self._res_names[:]

    def get_res_ids(self):
        """Get a list of residue IDs for selected atoms
        """
        return self._res_ids[:]

    def get_seg_indexes(self):
        """Get a list of segment indexes for selected atoms
        
        Note that segment index starts from 0
        """
        return self._seg_indexes[:]

    def get_seg_ids(self):
        """Get a list of segment IDs for selected atoms
        """
        return self._seg_ids[:]

    def get_atom_types(self):
        """Get a list of atom names for selected atoms
        """
        return self._atom_types[:]


class SelectAtomsIterator:
    def __init__(self, select_atoms):
        self._select_atoms = select_atoms
        self._index = 0

    def __next__(self):
        if self._index < len(self._select_atoms):
            is_sel_i = self._select_atoms.is_selected(self._index)
            self._index += 1
            return is_sel_i

        raise StopIteration
