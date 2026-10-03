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
import re
import string
import typing
import weakref

import numpy
import numpy as np  # alias for convenience

import pycharmm.psf as psf
import pycharmm.select as select
import pycharmm.atom_info as atom_info


# Names queued by GC finalizers for deferred removal from CHARMM's stored-
# selection table. See _unstore_on_gc / _drain_pending_unstores.
_pending_gc_unstores = []


def _unstore_on_gc(name):
    """weakref.finalize callback for a garbage-collected stored SelectAtoms.

    This can run during cyclic GC at an arbitrary allocation point, possibly
    while an in-flight CHARMM operation is walking the shared stored-selection
    table.  So it must NOT touch that table here -- it only queues the name
    (a GIL-atomic list append).  The actual delete happens later, from a
    top-level context, in _drain_pending_unstores().

    Parameters
    ----------
    name : str
        Name of the stored selection to remove on the next drain.
    """
    _pending_gc_unstores.append(name)


def _drain_pending_unstores():
    """Remove any selections queued by GC finalizers.

    Called from top-level SelectAtoms table operations (store/unstore), where
    no CHARMM table walk is in flight, so deleting is safe.  Best-effort:
    guarded on the library still being initialized, and each delete is
    independently protected.  Names are stored upper-cased in CHARMM, so
    probe/delete with the upper-cased name.
    """
    if not _pending_gc_unstores:
        return
    try:
        from pycharmm.loader import is_initialized
        if not is_initialized():
            _pending_gc_unstores.clear()
            return
    except Exception:
        return
    while _pending_gc_unstores:
        name = _pending_gc_unstores.pop()
        try:
            if select.find(name.upper()) > 0:
                select.delete_stored_selection(name.upper())
        except Exception:
            pass


# =============================================================================
# String-based selection parser (MDAnalysis-compatible syntax)
# =============================================================================

def parse_selection(selection_string: str) -> 'SelectAtoms':
    """Parse an MDAnalysis-compatible selection string.

    Parameters
    ----------
    selection_string : str
        Selection string in MDAnalysis-compatible format.

    Returns
    -------
    SelectAtoms
        Selection object based on the parsed string.

    Examples
    --------
    >>> sel = parse_selection('segid PROA')
    >>> sel = parse_selection('segid PROA and resid 1')
    >>> sel = parse_selection('resname ALA or resname GLY')
    >>> sel = parse_selection('segid PROA and (resid 1 or resid 2)')
    >>> sel = parse_selection('not hydrogens')
    >>> sel = parse_selection('name CA CB')  # Multiple values = OR

    Supported keywords:
    - segid <value(s)>: Select by segment ID
    - resid <value(s)>: Select by residue ID
    - resname <value(s)>: Select by residue name
    - name <value(s)>: Select by atom name (atom type)
    - type <value(s)>: Select by chemical type
    - all: Select all atoms
    - protein: Select protein atoms (common residues)
    - backbone: Select backbone atoms (N, CA, C, O)
    - water: Select water molecules (TIP3, HOH, WAT)
    - ions: Select common ions (SOD, CLA, POT, etc.)
    - hydrogens: Select hydrogen atoms

    Operators:
    - and: Intersection of selections
    - or: Union of selections
    - not: Negation of selection
    - (): Grouping
    """
    return _SelectionParser(selection_string).parse()


class _SelectionParser:
    """Internal parser for selection strings."""

    # Token patterns
    KEYWORDS = {
        'segid', 'resid', 'resname', 'name', 'type',
        'all', 'protein', 'backbone', 'water', 'ions', 'hydrogens',
        'and', 'or', 'not'
    }

    # Preset residue lists
    PROTEIN_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET',
        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }

    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OT1', 'OT2', 'OXT'}

    WATER_RESIDUES = {'TIP3', 'HOH', 'WAT', 'TIP4', 'SPC'}

    ION_RESIDUES = {'SOD', 'CLA', 'POT', 'MG', 'CA', 'ZN', 'NA', 'CL', 'K'}

    def __init__(self, selection_string: str):
        self.original = selection_string
        self.tokens = self._tokenize(selection_string)
        self.pos = 0

    def _tokenize(self, s: str) -> list:
        """Tokenize the selection string."""
        # Replace operators with spaced versions for easier parsing
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        tokens = s.split()
        return tokens

    def _peek(self) -> str:
        """Peek at current token without consuming."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].lower()
        return ''

    def _consume(self) -> str:
        """Consume and return current token."""
        token = self.tokens[self.pos] if self.pos < len(self.tokens) else ''
        self.pos += 1
        return token

    def _expect(self, expected: str):
        """Consume and verify expected token."""
        token = self._consume()
        if token.lower() != expected.lower():
            raise ValueError(f"Expected '{expected}', got '{token}'")

    def _get_values(self) -> list:
        """Get one or more values until an operator or end."""
        values = []
        while self.pos < len(self.tokens):
            token = self._peek()
            if token in ('and', 'or', 'not', '(', ')') or token in self.KEYWORDS:
                break
            values.append(self._consume())
        return values

    def parse(self) -> 'SelectAtoms':
        """Parse the selection string and return SelectAtoms."""
        if not self.tokens:
            return SelectAtoms()
        result = self._parse_or()
        return result

    def _parse_or(self) -> 'SelectAtoms':
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and()
        while self._peek() == 'or':
            self._consume()  # consume 'or'
            right = self._parse_and()
            left = left | right
        return left

    def _parse_and(self) -> 'SelectAtoms':
        """Parse AND expressions."""
        left = self._parse_not()
        while self._peek() == 'and':
            self._consume()  # consume 'and'
            right = self._parse_not()
            left = left & right
        return left

    def _parse_not(self) -> 'SelectAtoms':
        """Parse NOT expressions."""
        if self._peek() == 'not':
            self._consume()  # consume 'not'
            operand = self._parse_not()  # NOT is right-associative
            return ~operand
        return self._parse_primary()

    def _parse_primary(self) -> 'SelectAtoms':
        """Parse primary expressions (keywords, parentheses)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_or()
            self._expect(')')
            return result

        if token == 'segid':
            self._consume()
            values = self._get_values()
            return SelectAtoms(segid=values if len(values) > 1 else values[0] if values else '', update=False)

        if token == 'resid':
            self._consume()
            values = self._get_values()
            return SelectAtoms(resid=values if len(values) > 1 else values[0] if values else '', update=False)

        if token == 'resname':
            self._consume()
            values = self._get_values()
            return SelectAtoms(resname=values if len(values) > 1 else values[0] if values else '', update=False)

        if token == 'name':
            self._consume()
            values = self._get_values()
            return SelectAtoms(name=values if len(values) > 1 else values[0] if values else '', update=False)

        if token == 'type':
            self._consume()
            values = self._get_values()
            return SelectAtoms(type=values if len(values) > 1 else values[0] if values else '', update=False)

        if token == 'all':
            self._consume()
            return SelectAtoms(select_all=True, update=False)

        if token == 'protein':
            self._consume()
            return SelectAtoms(resname=list(self.PROTEIN_RESIDUES), update=False)

        if token == 'backbone':
            self._consume()
            protein = SelectAtoms(resname=list(self.PROTEIN_RESIDUES), update=False)
            bb_atoms = SelectAtoms(name=list(self.BACKBONE_ATOMS), update=False)
            return protein & bb_atoms

        if token == 'water':
            self._consume()
            return SelectAtoms(resname=list(self.WATER_RESIDUES), update=False)

        if token == 'ions':
            self._consume()
            return SelectAtoms(resname=list(self.ION_RESIDUES), update=False)

        if token == 'hydrogens':
            self._consume()
            return SelectAtoms(hydrogens=True, update=False)

        # Unknown token - try to treat as a value
        if token and token not in ('and', 'or', 'not', '(', ')'):
            raise ValueError(f"Unknown selection keyword: '{token}'")

        return SelectAtoms(update=False)


# =============================================================================
# Preset selection functions
# =============================================================================

# Preset residue/atom lists (shared with parser)
PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU', 'LYS', 'MET',
    'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OT1', 'OT2', 'OXT'}

WATER_RESIDUES = {'TIP3', 'HOH', 'WAT', 'TIP4', 'SPC'}

ION_RESIDUES = {'SOD', 'CLA', 'POT', 'MG', 'ZN', 'NA', 'CL', 'K'}

NUCLEIC_RESIDUES = {
    'ADE', 'THY', 'GUA', 'CYT', 'URA',  # Standard
    'DA', 'DT', 'DG', 'DC',  # DNA
    'A', 'U', 'G', 'C',  # RNA
}


def protein(update: bool = True) -> 'SelectAtoms':
    """Select all protein atoms.

    Returns
    -------
    SelectAtoms
        Selection containing all atoms in protein residues.

    Example
    -------
    >>> prot = protein()
    >>> ca_atoms = prot & SelectAtoms(name='CA')
    """
    return SelectAtoms(resname=list(PROTEIN_RESIDUES), update=update)


def backbone(update: bool = True) -> 'SelectAtoms':
    """Select protein backbone atoms (N, CA, C, O).

    Returns
    -------
    SelectAtoms
        Selection containing backbone atoms (N, CA, C, O) in protein residues.

    Example
    -------
    >>> bb = backbone()
    """
    prot = SelectAtoms(resname=list(PROTEIN_RESIDUES), update=False)
    bb = SelectAtoms(name=list(BACKBONE_ATOMS), update=False)
    result = prot & bb
    if update:
        result._do_update = True
        result._update()
    return result


def water(update: bool = True) -> 'SelectAtoms':
    """Select all water molecules.

    Returns
    -------
    SelectAtoms
        Selection containing all water molecules (TIP3, HOH, WAT, etc.).

    Example
    -------
    >>> wat = water()
    """
    return SelectAtoms(resname=list(WATER_RESIDUES), update=update)


def ions(update: bool = True) -> 'SelectAtoms':
    """Select all ion atoms.

    Returns
    -------
    SelectAtoms
        Selection containing common ions (SOD, CLA, POT, etc.).

    Example
    -------
    >>> ion_sel = ions()
    """
    return SelectAtoms(resname=list(ION_RESIDUES), update=update)


def nucleic(update: bool = True) -> 'SelectAtoms':
    """Select all nucleic acid atoms (DNA/RNA).

    Returns
    -------
    SelectAtoms
        Selection containing nucleic acid residues.

    Example
    -------
    >>> dna_rna = nucleic()
    """
    return SelectAtoms(resname=list(NUCLEIC_RESIDUES), update=update)


def sidechain(update: bool = True) -> 'SelectAtoms':
    """Select protein sidechain atoms (non-backbone).

    Returns
    -------
    SelectAtoms
        Selection containing sidechain atoms in protein residues.

    Example
    -------
    >>> sc = sidechain()
    """
    prot = SelectAtoms(resname=list(PROTEIN_RESIDUES), update=False)
    bb = SelectAtoms(name=list(BACKBONE_ATOMS), update=False)
    result = prot & (~bb)
    if update:
        result._do_update = True
        result._update()
    return result


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
                 seg_id='', segid='',
                 res_id='', resid='',
                 res_name='', resname='',
                 atom_nums=None,
                 atom_type='', name='',
                 chem_type='', type='',
                 initials=False,
                 lonepairs=False,
                 hydrogens=False,
                 update=True):
        """
        Parameters
        ----------
        selection : bool tuple or numpy array
            A boolean tuple/array whose length is the number of atoms
            in the system. True if the corresponding atom is selected.
        select_all : bool
            True for selecting all atoms
        seg_id, segid : str or list of str
            segment ID of the selection (segid is MDAnalysis-compatible alias).
            If list, selects atoms in any of the specified segments (OR).
        res_id, resid : str or list of str
            residue ID of the selection (resid is MDAnalysis-compatible alias).
            If list, selects atoms in any of the specified residues (OR).
        res_name, resname : str or list of str
            residue name of the selection (resname is MDAnalysis-compatible alias).
            If list, selects atoms in any of the specified residue types (OR).
        atom_nums : int or list of int
            atom index, i.e., atom number 
        atom_type, name : str or list of str
            atom names of the selection (name is MDAnalysis-compatible alias).
            If list, selects atoms with any of the specified names (OR).
        chem_type, type : str or list of str
            chemical type of the selection (type is MDAnalysis-compatible alias).
            If list, selects atoms with any of the specified types (OR).
        initials : bool
            True for selecting all atoms with known coordinates
        lonepairs : bool
            True for selecting all lone pairs
        hydrogens : bool
            True for selecting all hydrogen atoms
        update : bool
            True for updating the selection
            
        Note
        ----
        When multiple criteria are specified (seg_id, res_id, res_name, etc.),
        they are combined with AND logic. For example:
        SelectAtoms(seg_id='PROA', res_id='1') selects atoms in segment PROA AND residue 1.
        
        When a list is provided for a single parameter, those values are combined with OR:
        SelectAtoms(res_name=['ALA', 'GLY']) selects atoms in ALA OR GLY residues.
        
        MDAnalysis-compatible aliases are provided:
        - segid → seg_id
        - resid → res_id  
        - resname → res_name
        - name → atom_type
        - type → chem_type
        
        Use the | (or) and & (and) operators to combine SelectAtoms objects:
        SelectAtoms(seg_id='PROA') | SelectAtoms(seg_id='PROB')  # OR
        SelectAtoms(seg_id='PROA') & SelectAtoms(atom_type='CA')  # AND
        """
        self._name = ''
        self._stored = False
        self._finalizer = None
        self._do_update = update
        self._selection = None
        self._properties_computed = False
        self._atom_indexes = []
        self._n_selected = 0
        n_atoms = psf.get_natom()

        # Handle MDAnalysis-compatible aliases (alias takes precedence if both specified)
        _seg_id = segid or seg_id
        _res_id = resid or res_id
        _res_name = resname or res_name
        _atom_type = name or atom_type
        _chem_type = type or chem_type
        
        def _make_list_criterion(values, batch_func):
            """Select by a single value or any of a list of values.

            Both cases go through the batched (set-membership, single-pass)
            selector, so e.g. a 22-residue protein selection is one scan
            rather than 22 scans OR-ed together.
            """
            if isinstance(values, (list, tuple)):
                if len(values) == 0:
                    return None
                return batch_func(list(values))
            elif values:
                return batch_func([values])
            return None

        # Collect criteria to determine if we should use AND logic
        criteria = []

        crit = _make_list_criterion(_seg_id, select.by_segment_ids)
        if crit is not None:
            criteria.append(crit)

        crit = _make_list_criterion(_res_id, select.by_residue_ids)
        if crit is not None:
            criteria.append(crit)

        crit = _make_list_criterion(_res_name, select.by_residue_names)
        if crit is not None:
            criteria.append(crit)
            
        if atom_nums is not None:
            if isinstance(atom_nums, int):
                atom_nums = [atom_nums]
            atom_sel = select.none_selection(n_atoms)
            for num in atom_nums:
                if 0 <= num < n_atoms:
                    atom_sel[num] = True
            criteria.append(atom_sel)
            
        crit = _make_list_criterion(_atom_type, select.by_atom_types)
        if crit is not None:
            criteria.append(crit)

        crit = _make_list_criterion(_chem_type, select.by_chem_types)
        if crit is not None:
            criteria.append(crit)
            
        if initials:
            criteria.append(select.initial())
        if lonepairs:
            criteria.append(select.lone())
        if hydrogens:
            criteria.append(select.hydrogen())
        
        # Determine base selection
        if selection is not None:
            # User provided explicit selection - use it as base
            base = select._ensure_numpy(selection)
        elif select_all:
            base = select.all_selection(n_atoms)
        elif criteria:
            # We have criteria but no explicit selection - start with all atoms
            base = select.all_selection(n_atoms)
        else:
            # No criteria and no selection - empty selection
            base = select.none_selection(n_atoms)
        
        # Apply criteria with AND logic
        if criteria:
            result = base
            for crit in criteria:
                result = select.and_selection(result, crit)
            self.set_selection(result)
        else:
            self.set_selection(base)

    def _select_atom(self, atom_i):
        old_val = self[atom_i]
        self.set_selection(select.by_atom_inds((atom_i,), self.get_selection()))
        return old_val

    def _deselect_atom(self, atom_i):
        old_val = self[atom_i]
        new_sel = tuple(sel if not ind == atom_i else False
                        for ind, sel in enumerate(self.get_selection()))
        self.set_selection(new_sel)
        return old_val

    def get_selection(self) -> np.ndarray:
        """
        For a pycharmm selection, return a boolean numpy array, whose length
        is the number of atoms in the system.

        True if an atom is in the selection, otherwise False.
        
        Note
        ----
        Returns numpy array for performance. Use as_tuple() for tuple.
        """
        return self._selection

    def set_selection(self, selection: select.Selection):
        """
        For a pycharmm selection, update it based on the input boolean tuple or array.

        Parameters
        ----------
        selection: boolean tuple or numpy array
            Length of the tuple/array is equal to the number of atoms in the system

            True if an atom is in the selection, otherwise False.

        """
        # Store internally as numpy array for performance
        if isinstance(selection, np.ndarray):
            self._selection = selection
        else:
            self._selection = np.asarray(selection, dtype=bool) if selection else np.array([], dtype=bool)

        # Always compute atom indices (cheap - just np.nonzero)
        # This ensures get_atom_indexes() works even with update=False
        self._atom_indexes = np.nonzero(self._selection)[0].tolist() if len(self._selection) > 0 else []
        self._n_selected = len(self._atom_indexes)

        # Mark properties as stale (will be computed lazily on first access)
        self._properties_computed = False

        if self._do_update:
            self._update()

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
        return self._selection.tolist() if isinstance(self._selection, numpy.ndarray) else list(self._selection)

    def as_tuple(self) -> tuple:
        """Convert selection to tuple for backward compatibility.
        
        Returns
        -------
        tuple
            Boolean tuple where True indicates atom is selected.
        """
        if isinstance(self._selection, numpy.ndarray):
            return tuple(self._selection)
        return tuple(self._selection) if self._selection else ()

    def __array__(self, dtype=None, copy=None) -> numpy.ndarray:
        """NumPy array protocol.

        NumPy may call this with ``dtype`` and (NumPy>=2) a ``copy=`` keyword.
        """
        arr = self._selection if isinstance(self._selection, numpy.ndarray) else numpy.array(list(self), dtype=bool)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy is True:
            return arr.copy()
        return arr

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
        # Use update=False to defer expensive property computation
        # Properties are computed lazily on first access
        return SelectAtoms(new_sel, update=False)

    def __or__(self, other):
        new_sel = select.or_selection(self.get_selection(),
                                      other.get_selection())
        return SelectAtoms(new_sel, update=False)

    def __invert__(self):
        new_sel = select.not_selection(self.get_selection())
        return SelectAtoms(new_sel, update=False)

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
        # Reclaim any slots freed by garbage-collected selections before we
        # add a new one (safe top-level context; see _drain_pending_unstores).
        _drain_pending_unstores()

        max_name = select.get_max_name()
        if name:
            self._name = name[:max_name]

        if not self._name:
            self._name = _generate_script_name()

        if len(self._name) > max_name:
            self._name = self._name[:max_name]

        select.store_selection(self._name, self.get_selection())
        self._stored = True

        # (Re)arm a garbage-collection finalizer so a stored selection that
        # is dropped without an explicit unstore() still frees its slot in
        # CHARMM's (now growable) stored-selection table. Cancel any prior
        # finalizer first in case the name changed across store() calls.
        if self._finalizer is not None:
            self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _unstore_on_gc, self._name)
        self._finalizer.atexit = False

        return self._name

    def unstore(self):
        """
        Unstore/remove the named pycharmm selection in CHARMM

        Returns
        -------
        was_stored : old store status
        """
        _drain_pending_unstores()

        was_stored = self.is_stored()
        if was_stored:
            select.delete_stored_selection(self._name.upper())
            self._stored = False

        # Explicit unstore supersedes the GC finalizer; cancel it so the
        # name is not deleted a second time when this object is collected.
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

        return was_stored

    def __enter__(self):
        self.store()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unstore()

    def _update(self):
        """Compute atom properties for selected atoms.

        This is called automatically when update=True, or lazily on first
        property access when update=False.
        """
        inds = self._atom_indexes
        # Use fast vectorized functions (O(n_selected) instead of O(n_atoms * n_res))
        self._chem_types = atom_info.get_chem_types_fast(inds)
        self._res_indexes = atom_info.get_res_indexes_fast(inds)
        self._res_names = atom_info.get_res_names_fast(inds)
        self._res_ids = atom_info.get_res_ids_fast(inds)
        self._seg_indexes = atom_info.get_seg_indexes_fast(inds)
        self._seg_ids = atom_info.get_seg_ids_fast(inds)
        self._atom_types = atom_info.get_atom_types_fast(inds)
        self._properties_computed = True
        if self.is_stored():
            self.store()

        return self

    def _ensure_properties(self):
        """Ensure atom properties are computed (lazy initialization)."""
        if not getattr(self, '_properties_computed', False):
            self._update()

    def get_atom_indexes(self):
        """Get a list of atom indexes for selected atoms.

        Note that atom index starts from 0
        """
        return self._atom_indexes[:]

    # Aliases for get_atom_indexes (grammatically correct alternatives)
    get_atom_indices = get_atom_indexes
    get_indices = get_atom_indexes

    def get_n_selected(self):
        """ Get number of selected atoms
        """
        return self._n_selected

    def get_chem_types(self):
        """Get a list of chemical types (based on the topology file)
        for selected atoms
        """
        self._ensure_properties()
        return self._chem_types[:]

    def get_res_indexes(self):
        """Get a list of residue indexes for selected atoms.

        Note that residue index starts from 0
        """
        self._ensure_properties()
        return self._res_indexes[:]

    def get_res_names(self):
        """Get a list of residue names for selected atoms
        """
        self._ensure_properties()
        return self._res_names[:]

    def get_res_ids(self):
        """Get a list of residue IDs for selected atoms
        """
        self._ensure_properties()
        return self._res_ids[:]

    def get_seg_indexes(self):
        """Get a list of segment indexes for selected atoms

        Note that segment index starts from 0
        """
        self._ensure_properties()
        return self._seg_indexes[:]

    def get_seg_ids(self):
        """Get a list of segment IDs for selected atoms
        """
        self._ensure_properties()
        return self._seg_ids[:]

    def get_atom_types(self):
        """Get a list of atom names for selected atoms
        """
        self._ensure_properties()
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
