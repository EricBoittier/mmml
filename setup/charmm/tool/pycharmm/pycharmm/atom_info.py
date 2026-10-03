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
            raise ValueError(msg.format(i, n_atoms))

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

    atom_indexes = selection.get_atom_indexes()
    all_pos = coor.get_positions()
    all_weights = coor.get_weights()
    atom_weights = [all_weights[i] for i in atom_indexes]
    atom_table = pandas.DataFrame({
        'index': atom_indexes,
        'type': selection.get_atom_types(),
        'residue_number': selection.get_res_indexes(),
        'residue_name': selection.get_res_names(),
        'residue_id': selection.get_res_ids(),
        'segment_number': selection.get_seg_indexes(),
        'segment_id': selection.get_seg_ids(),
        'x': [all_pos.iloc[i, 0] for i in atom_indexes],
        'y': [all_pos.iloc[i, 1] for i in atom_indexes],
        'z': [all_pos.iloc[i, 2] for i in atom_indexes],
        'w': atom_weights})

    return atom_table


# =============================================================================
# Fast Atom Lookup Cache
# =============================================================================
# This provides fast Python-side atom lookups without calling CHARMM's
# selection mechanism for each query. Useful for bulk operations like
# setting up many NOE restraints.

class AtomLookupCache:
    """Fast atom lookup cache for bulk selection operations.

    Caches atom info in numpy arrays and provides fast filtering methods.
    Auto-invalidates when the number of atoms changes.

    Examples
    --------
    >>> from pycharmm.atom_info import AtomLookupCache
    >>> cache = AtomLookupCache()
    >>>
    >>> # Find all OH2 atoms in RESV segment
    >>> indices = cache.find_atoms(seg_id='RESV', atom_type='OH2')
    >>>
    >>> # Find specific atom by seg_id + res_id + atom_type
    >>> idx = cache.find_atoms(seg_id='RESV', res_id='1', atom_type='OH2')
    >>>
    >>> # Get atoms grouped by residue
    >>> for res_id, atom_indices in cache.iter_by_residue(seg_id='RESV', atom_type='OH2'):
    ...     print(f"Residue {res_id}: atom {atom_indices[0]}")
    """

    def __init__(self):
        self._natom = None
        self._nseg = None            # number of segments (for change detection)
        self._nres = None            # number of residues (for change detection)
        self._fingerprint = None     # structure fingerprint for detecting reordering
        self._atom_types = None      # numpy array of atom type strings
        self._seg_indices = None     # numpy array of segment indices per atom
        self._res_indices = None     # numpy array of residue indices per atom
        self._seg_ids = None         # list of segment ID strings
        self._res_ids = None         # list of residue ID strings
        # O(1) lookup dicts
        self._seg_id_to_idx = None   # dict: seg_id -> seg_index
        self._res_id_to_idx = None   # dict: res_id -> res_index (first occurrence)
        # Pre-computed boolean masks for segments (most common query)
        self._seg_masks = None       # dict: seg_id -> numpy bool array
        # Per-atom residue IDs for fast residue selection
        self._per_atom_res_ids = None  # numpy array of residue ID strings per atom

    def _compute_fingerprint(self):
        """Compute a fingerprint of the PSF structure for change detection."""
        # Use segment IDs as fingerprint (fast to get, sensitive to changes)
        seg_ids = psf.get_segid()
        return tuple(seg_ids) if seg_ids else ()

    def _ensure_cache(self):
        """Rebuild cache if needed (first call or structure changed).

        Checks multiple indicators to detect structural changes:
        - Number of atoms (catches add/delete)
        - Number of segments (catches segment operations)
        - Number of residues (catches residue operations)
        - Segment IDs fingerprint (catches reordering)
        """
        import numpy as np

        current_natom = psf.get_natom()
        current_nseg = psf.get_nseg()
        current_nres = psf.get_nres()
        current_fingerprint = self._compute_fingerprint()

        # Check if cache is still valid
        if (self._natom == current_natom and
            self._nseg == current_nseg and
            self._nres == current_nres and
            self._fingerprint == current_fingerprint and
            self._atom_types is not None):
            return  # Cache is valid

        # Rebuild cache - store structure identifiers
        self._natom = current_natom
        self._nseg = current_nseg
        self._nres = current_nres
        self._fingerprint = current_fingerprint

        if current_natom == 0:
            self._atom_types = np.array([], dtype=object)
            self._seg_indices = np.array([], dtype=np.int32)
            self._res_indices = np.array([], dtype=np.int32)
            self._seg_ids = []
            self._res_ids = []
            self._seg_id_to_idx = {}
            self._res_id_to_idx = {}
            self._seg_masks = {}
            self._per_atom_res_ids = np.array([], dtype=object)
            return

        # Get all atom types (IUPAC names like "OH2", "CA", etc.)
        self._atom_types = np.array(psf.get_atype(), dtype=object)

        # Get atom→segment and atom→residue mappings
        self._seg_indices = np.array(atom_to_seg(), dtype=np.int32)
        self._res_indices = np.array(atom_to_res(), dtype=np.int32)

        # Get segment and residue ID strings
        self._seg_ids = psf.get_segid()
        self._res_ids = psf.get_resid()

        # Build O(1) lookup dicts
        self._seg_id_to_idx = {sid: i for i, sid in enumerate(self._seg_ids)}
        self._res_id_to_idx = {rid: i for i, rid in enumerate(self._res_ids)}

        # Pre-compute segment masks (very common query pattern)
        self._seg_masks = {}
        for seg_id, seg_idx in self._seg_id_to_idx.items():
            self._seg_masks[seg_id] = (self._seg_indices == seg_idx)

        # Build per-atom residue ID array for fast residue selection
        # This maps each atom to its residue ID string
        self._per_atom_res_ids = np.empty(current_natom, dtype=object)
        for i in range(current_natom):
            res_idx = self._res_indices[i]
            if 0 <= res_idx < len(self._res_ids):
                self._per_atom_res_ids[i] = self._res_ids[res_idx]
            else:
                self._per_atom_res_ids[i] = ""

    def invalidate(self):
        """Force cache invalidation (call after adding/deleting atoms)."""
        self._natom = None

    def find_atoms(self, seg_id=None, res_id=None, atom_type=None):
        """Find atom indices matching criteria.

        Parameters
        ----------
        seg_id : str, optional
            Segment ID to match (e.g., 'RESV', 'PROA')
        res_id : str, optional
            Residue ID to match (e.g., '1', '2')
        atom_type : str, optional
            Atom type/name to match (e.g., 'OH2', 'CA', 'N')

        Returns
        -------
        numpy.ndarray
            1-based atom indices matching all criteria
        """
        import numpy as np

        self._ensure_cache()

        if self._natom == 0:
            return np.array([], dtype=np.int32)

        # Start with all atoms selected (or use pre-computed segment mask)
        if seg_id is not None:
            if seg_id not in self._seg_masks:
                return np.array([], dtype=np.int32)  # Segment not found
            mask = self._seg_masks[seg_id].copy()
        else:
            mask = np.ones(self._natom, dtype=bool)

        # Filter by residue ID using O(1) dict lookup
        if res_id is not None:
            res_idx = self._res_id_to_idx.get(res_id)
            if res_idx is None:
                return np.array([], dtype=np.int32)  # Residue not found
            mask &= (self._res_indices == res_idx)

        # Filter by atom type
        if atom_type is not None:
            mask &= (self._atom_types == atom_type)

        # Return 1-based indices
        return np.where(mask)[0] + 1

    def find_atoms_in_segment(self, seg_id, atom_type=None):
        """Find all atoms in a segment, optionally filtered by type.

        Parameters
        ----------
        seg_id : str
            Segment ID to search in
        atom_type : str, optional
            Atom type to filter by

        Returns
        -------
        dict
            Mapping of res_id → list of 1-based atom indices
        """
        import numpy as np
        from collections import defaultdict

        self._ensure_cache()

        if self._natom == 0:
            return {}

        try:
            seg_idx = self._seg_ids.index(seg_id)
        except ValueError:
            return {}

        # Find atoms in this segment
        mask = (self._seg_indices == seg_idx)
        if atom_type is not None:
            mask &= (self._atom_types == atom_type)

        atom_indices_0based = np.where(mask)[0]

        # Group by residue
        result = defaultdict(list)
        for atom_idx in atom_indices_0based:
            res_idx = self._res_indices[atom_idx]
            res_id = self._res_ids[res_idx]
            result[res_id].append(atom_idx + 1)  # 1-based

        return dict(result)

    def iter_by_residue(self, seg_id, atom_type=None):
        """Iterate over atoms grouped by residue within a segment.

        Parameters
        ----------
        seg_id : str
            Segment ID to search in
        atom_type : str, optional
            Atom type to filter by

        Yields
        ------
        tuple
            (res_id, list of 1-based atom indices)
        """
        atoms_by_res = self.find_atoms_in_segment(seg_id, atom_type)
        # Sort by residue ID numerically if possible
        try:
            sorted_res_ids = sorted(atoms_by_res.keys(), key=int)
        except ValueError:
            sorted_res_ids = sorted(atoms_by_res.keys())

        for res_id in sorted_res_ids:
            yield res_id, atoms_by_res[res_id]

    def get_n_atoms(self):
        """Get the number of atoms in the system."""
        self._ensure_cache()
        return self._natom


# Global cache instance for convenience
_atom_cache = None


def get_atom_cache():
    """Get the global atom lookup cache instance.

    Returns
    -------
    AtomLookupCache
        The shared cache instance

    Examples
    --------
    >>> from pycharmm.atom_info import get_atom_cache
    >>> cache = get_atom_cache()
    >>> indices = cache.find_atoms(seg_id='RESV', atom_type='OH2')
    """
    global _atom_cache
    if _atom_cache is None:
        _atom_cache = AtomLookupCache()
    return _atom_cache


def invalidate_atom_cache():
    """Invalidate the global atom cache (call after adding/deleting atoms)."""
    global _atom_cache
    if _atom_cache is not None:
        _atom_cache.invalidate()


# =============================================================================
# Fast Vectorized Atom Info Functions (use AtomLookupCache)
# =============================================================================
# These functions provide O(1) or O(n_selected) performance vs O(n_atoms * n_res)
# for the original functions. Use these for bulk operations.

import numpy as np


def get_res_indexes_fast(atom_indices):
    """Get the residue index for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[int]
        List of residue indices, one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    if cache._res_indices is None or len(cache._res_indices) == 0:
        return []

    # Convert to numpy array for vectorized indexing
    indices = np.asarray(atom_indices, dtype=np.intp)
    if len(indices) == 0:
        return []

    # Bounds check and vectorized lookup
    n_atoms = len(cache._res_indices)
    valid_mask = (indices >= 0) & (indices < n_atoms)
    if not np.all(valid_mask):
        # Fallback to per-element checking for error reporting
        for i in indices[~valid_mask]:
            raise ValueError(f'No residue available for atom index {i}')

    return cache._res_indices[indices].tolist()


def get_seg_indexes_fast(atom_indices):
    """Get the segment index for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[int]
        A list of segment indices, one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    if cache._seg_indices is None or len(cache._seg_indices) == 0:
        return []

    # Convert to numpy array for vectorized indexing
    indices = np.asarray(atom_indices, dtype=np.intp)
    if len(indices) == 0:
        return []

    # Bounds check and vectorized lookup
    n_atoms = len(cache._seg_indices)
    valid_mask = (indices >= 0) & (indices < n_atoms)
    if not np.all(valid_mask):
        for i in indices[~valid_mask]:
            raise ValueError(f'No segment available for atom index {i}')

    return cache._seg_indices[indices].tolist()


def get_res_names_fast(atom_indices):
    """Get the residue name for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[str]
        A list of residue names (str), one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    res_indices = get_res_indexes_fast(atom_indices)
    res = psf.get_res()

    if not res:
        return []

    # Vectorized lookup
    return [res[i] for i in res_indices]


def get_res_ids_fast(atom_indices):
    """Get the residue id for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[str]
        A list of residue ids (str), one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    res_indices = get_res_indexes_fast(atom_indices)
    rid = psf.get_resid()

    if not rid:
        return []

    return [rid[i] for i in res_indices]


def get_seg_ids_fast(atom_indices):
    """Get the segment id for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[str]
        A list of segment ids, one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    seg_indices = get_seg_indexes_fast(atom_indices)

    if cache._seg_ids is None or len(cache._seg_ids) == 0:
        return []

    return [cache._seg_ids[i] for i in seg_indices]


def get_atom_types_fast(atom_indices):
    """Get the atom name for each atom index (fast vectorized version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        A list/array of 0-based atom indices

    Returns
    -------
    list[str]
        a list of atom types (str), one for each atom index
    """
    if atom_indices is None:
        return []

    cache = get_atom_cache()
    cache._ensure_cache()

    if cache._atom_types is None or len(cache._atom_types) == 0:
        return []

    # Convert to numpy array for vectorized indexing
    indices = np.asarray(atom_indices, dtype=np.intp)
    if len(indices) == 0:
        return []

    n_atoms = len(cache._atom_types)
    valid_mask = (indices >= 0) & (indices < n_atoms)
    if not np.all(valid_mask):
        for i in indices[~valid_mask]:
            raise ValueError(f'atom index {i} >= number of atoms {n_atoms}')

    return cache._atom_types[indices].tolist()


def get_chem_types_fast(atom_indices):
    """Get the chemical types for each atom index (fast version).

    Parameters
    ----------
    atom_indices : list[int] or numpy.ndarray
        List/array of 0-based atom indices

    Returns
    -------
    list[str]
        A list of chem_type (str) for each atom index
    """
    if atom_indices is None:
        return []

    natc = param.get_natc()
    atc = param.get_atc()
    iac = psf.get_iac()
    n_atoms = psf.get_natom()

    if not atc or not iac:
        return []

    # Convert to numpy for vectorized operations
    indices = np.asarray(atom_indices, dtype=np.intp)
    if len(indices) == 0:
        return []

    # Bounds check
    if np.any(indices >= n_atoms) or np.any(indices < 0):
        for i in indices:
            if i >= n_atoms or i < 0:
                raise ValueError(f'atom index {i} >= number of atoms {n_atoms}')

    # Get iac codes for selected atoms
    iac_arr = np.array(iac, dtype=np.intp)
    selected_iac = iac_arr[indices]

    # Check iac bounds
    if np.any(selected_iac > natc):
        for i, idx in enumerate(indices):
            if iac[idx] > natc:
                raise ValueError(f'No chem type for atom {idx}')

    # Vectorized lookup
    return [atc[code] for code in selected_iac]
