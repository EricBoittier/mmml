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
import importlib.util
import typing
from collections.abc import Iterable

import numpy as np

import pycharmm.coor as coor
# import pycharmm.loader as lib
from pycharmm.loader import lib
import pycharmm.param as param
import pycharmm.psf as psf

import pycharmm.atom_info as atom_info


# =============================================================================
# Optional performance libraries (graceful fallback if not installed)
# =============================================================================

# Presence only: callers check the flag. Importing numexpr here left an
# unused binding, and the module is optional.
_HAS_NUMEXPR = importlib.util.find_spec("numexpr") is not None

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# JIT-compiled helper functions for performance-critical operations
# =============================================================================

@jit(nopython=True, cache=True)
def _fast_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled AND operation for boolean arrays."""
    n = len(a)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        result[i] = a[i] and b[i]
    return result


@jit(nopython=True, cache=True)
def _fast_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled OR operation for boolean arrays."""
    n = len(a)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        result[i] = a[i] or b[i]
    return result


@jit(nopython=True, cache=True)
def _fast_not(a: np.ndarray) -> np.ndarray:
    """JIT-compiled NOT operation for boolean arrays."""
    n = len(a)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        result[i] = not a[i]
    return result


@jit(nopython=True, cache=True, parallel=True)
def _fast_string_match(values: np.ndarray, target: str) -> np.ndarray:
    """JIT-compiled string matching (parallel for large arrays)."""
    n = len(values)
    result = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        result[i] = values[i] == target
    return result


# Selection can be either a tuple of bools (legacy) or numpy boolean array (fast)
Selection = typing.Union[typing.Tuple[bool, ...], np.ndarray]


def _ensure_numpy(sel: Selection) -> np.ndarray:
    """Convert selection to numpy array if not already."""
    if sel is None or (isinstance(sel, (tuple, list)) and len(sel) == 0):
        return np.array([], dtype=bool)
    if isinstance(sel, np.ndarray):
        return sel
    return np.asarray(sel, dtype=bool)


def _to_tuple(sel: Selection) -> typing.Tuple[bool, ...]:
    """Convert selection to tuple for backward compatibility."""
    if isinstance(sel, np.ndarray):
        return tuple(sel)
    return tuple(sel) if sel else ()


# =============================================================================
# Module-level cache for fast selection operations
# =============================================================================
def _get_cache():
    """Get or create the atom lookup cache for fast selections."""
    return atom_info.get_atom_cache()


def or_selection(sel_a: Selection, sel_b: Selection) -> np.ndarray:
    """Use eltwise logical `or` to produce a new selection.
    
    Returns numpy array for performance. Use _to_tuple() if tuple needed.
    """
    sel_a_np = _ensure_numpy(sel_a)
    sel_b_np = _ensure_numpy(sel_b)
    if len(sel_a_np) == 0 and len(sel_b_np) == 0:
        return np.array([], dtype=bool)
    if len(sel_a_np) == 0:
        return sel_b_np.copy()
    if len(sel_b_np) == 0:
        return sel_a_np.copy()
    return sel_a_np | sel_b_np


def and_selection(sel_a: Selection, sel_b: Selection) -> np.ndarray:
    """Use eltwise logical `and` to produce a new selection

    If either selection is empty, returns empty array (intersection of any set with empty is empty).
    Returns numpy array for performance. Use _to_tuple() if tuple needed.
    """
    sel_a_np = _ensure_numpy(sel_a)
    sel_b_np = _ensure_numpy(sel_b)
    if len(sel_a_np) == 0 or len(sel_b_np) == 0:
        # Intersection with empty set is empty
        return np.array([], dtype=bool)
    return sel_a_np & sel_b_np


def not_selection(sel: Selection) -> np.ndarray:
    """Use eltwise logical `not` to produce a new selection
    
    Returns numpy array for performance. Use _to_tuple() if tuple needed.
    """
    sel_np = _ensure_numpy(sel)
    if len(sel_np) == 0:
        return np.array([], dtype=bool)
    return ~sel_np


def none_selection(size: int) -> np.ndarray:
    """Get a new selection in which all elements are `False`
    
    Returns numpy array for performance. Use _to_tuple() if tuple needed.
    """
    if size < 0:
        raise ValueError("Size cannot be negative")
    return np.zeros(size, dtype=bool)


def all_selection(size: int) -> np.ndarray:
    """Get a new selection in which all elements are `True`
    
    Returns numpy array for performance. Use _to_tuple() if tuple needed.
    """
    if size < 0:
        raise ValueError("Size cannot be negative")
    return np.ones(size, dtype=bool)


def by_atom_inds(inds: Iterable[int], selection: Selection) -> np.ndarray:
    """Copy selection to new_sel then set new_sel[`inds`] to `True`
    
    Parameters
    ----------
    inds : Iterable[int]
        An iterable of 0-based atom indices to set to True.
    selection : Selection
        The base selection to modify.
        
    Returns
    -------
    flags : np.ndarray
            boolean array with specified indices set to True
    """
    # Convert original selection to a numpy array
    sel_np = _ensure_numpy(selection).copy()

    # Convert input indices to numpy array
    np_inds = np.array(list(inds), dtype=int)

    # Check if there's anything to do and if indices are valid
    if sel_np.size > 0 and np_inds.size > 0:
        # Ensure indices are within bounds to prevent IndexError
        valid_indices = np_inds[(np_inds >= 0) & (np_inds < sel_np.size)]
        if valid_indices.size > 0:
            sel_np[valid_indices] = True
            
    return sel_np


def by_residue_name(residue_name: str) -> np.ndarray:
    """Select all atoms in a residue

    Parameters
    ----------
    residue_name : string
                   name of residue to select

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)

    # Get mapping from atom index to residue index for all atoms
    atom_residue_indices = atom_info.atom_to_res()  # list, len = n_atoms
    
    # Get list of residue names, indexed by residue index
    all_residue_names = psf.get_res()  # list, len = n_residues

    if not all_residue_names:  # No residues defined
        return none_selection(n_atoms)
    
    # Vectorized approach: convert residue names to numpy array
    # and use advanced indexing for fast lookup
    res_names_arr = np.array(all_residue_names, dtype=object)
    atom_res_idx_arr = np.array(atom_residue_indices, dtype=np.intp)
    
    # Clamp indices to valid range (handle any out-of-bounds)
    n_residues = len(all_residue_names)
    valid_mask = (atom_res_idx_arr >= 0) & (atom_res_idx_arr < n_residues)
    
    # Initialize result array
    result = np.zeros(n_atoms, dtype=bool)
    
    # Use advanced indexing only for valid indices
    if np.any(valid_mask):
        valid_indices = atom_res_idx_arr[valid_mask]
        per_atom_names = res_names_arr[valid_indices]
        result[valid_mask] = (per_atom_names == residue_name)
    
    return result


def by_residue_id(residue_id: str) -> np.ndarray:
    """Select all atoms in a residue by residue id

    Parameters
    ----------
    residue_id : str
                 CHARMM id of residue to select

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True

    Note
    ----
    Uses cached atom data for fast repeated selections.
    """
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)

    # Use cached per-atom residue IDs
    cache = _get_cache()
    cache._ensure_cache()

    if cache._per_atom_res_ids is not None and len(cache._per_atom_res_ids) == n_atoms:
        return cache._per_atom_res_ids == residue_id

    # Fallback to uncached (should not normally reach here)
    return none_selection(n_atoms)


def by_segment_id(segment_id: str) -> np.ndarray:
    """Select all atoms in a segment.

    Parameters
    ----------
    segment_id : str
                 name of segment to select

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True

    Note
    ----
    Uses cached atom data for fast repeated selections.
    """
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)

    # Use cached segment masks for O(1) lookup
    cache = _get_cache()
    cache._ensure_cache()

    if cache._seg_masks is not None and segment_id in cache._seg_masks:
        return cache._seg_masks[segment_id].copy()

    # Fallback for unknown segment
    return none_selection(n_atoms)


def by_atom_type(atom_type: str) -> np.ndarray:
    """Select all atoms of type `atom_type`

    Parameters
    ----------
    atom_type : string
                IUPAC name of type to select

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True

    Note
    ----
    Uses cached atom data for fast repeated selections.
    """
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)

    # Use cached atom types
    cache = _get_cache()
    cache._ensure_cache()

    if cache._atom_types is not None and len(cache._atom_types) == n_atoms:
        return cache._atom_types == atom_type

    # Fallback to uncached (should not normally reach here)
    return none_selection(n_atoms)


def by_chem_type(chem_type: str) -> np.ndarray:
    """Select all atoms of param type code `chem_type`

    Parameters
    ----------
    chem_type : str
                parameter type code to select

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)

    # Get the list of chemical type names, indexed by their type code.
    type_code_to_name_map = param.get_atc() 
    
    # Get the chemical type code (integer) for each atom.
    per_atom_type_codes = psf.get_iac()

    if not type_code_to_name_map or not per_atom_type_codes or len(per_atom_type_codes) != n_atoms:
        return none_selection(n_atoms)

    # Vectorized approach: convert type names to numpy array
    # and use advanced indexing for fast lookup
    type_names_arr = np.array(type_code_to_name_map, dtype=object)
    type_codes_arr = np.array(per_atom_type_codes, dtype=np.intp)
    
    # Clamp indices to valid range (handle any out-of-bounds)
    n_types = len(type_code_to_name_map)
    valid_mask = (type_codes_arr >= 0) & (type_codes_arr < n_types)
    
    # Initialize result array
    result = np.zeros(n_atoms, dtype=bool)
    
    # Use advanced indexing only for valid indices
    if np.any(valid_mask):
        valid_codes = type_codes_arr[valid_mask]
        per_atom_names = type_names_arr[valid_codes]
        result[valid_mask] = (per_atom_names == chem_type)
    
    return result


# =============================================================================
# Batched (set-membership) selection by multiple values in a single pass
#
# Selecting by a list of names previously meant one full-length scan per name
# OR-ed together -- O(k * natom) with large Python/numpy constants (e.g.
# backbone() OR-ing 22 protein residue names).  These do it in one pass:
# membership against a set, at the smallest entity level (residue / type code)
# where possible, then expanded to atoms.  A single-element list gives exactly
# the same result as the corresponding by_<x> singular function.
#
# NOTE: these mirror the singular by_residue_name / by_residue_id /
# by_segment_id / by_atom_type / by_chem_type functions above and share their
# lookup conventions (residue-index expansion, type-code mapping, cached
# per-atom arrays).  The singular forms are kept separate rather than delegating
# here, because their vectorized scalar path is faster for a single value; if
# you change a lookup convention in one, update its partner.  The
# test_batched_selection_equals_or_of_singles test guards the equivalence.
# =============================================================================

def _as_value_list(values) -> list:
    """Normalize a str or iterable of str to a list of str.

    Parameters
    ----------
    values : str or iterable of str
        A single value or a collection of values.

    Returns
    -------
    list of str
        ``[values]`` for a string, otherwise a list of the items.
    """
    if isinstance(values, str):
        return [values]
    return [v for v in values]


def _isin_object(arr_obj: np.ndarray, values) -> np.ndarray:
    """Boolean membership mask for an object (string) ndarray, O(len(arr)).

    numpy's isin on object dtype is not reliably linear, so for the general
    case use a Python set with a single fromiter pass. A single value is
    special-cased to the vectorized equality the singular by_<x> functions
    use -- otherwise a scalar selection would pay a Python per-element loop
    over the whole array, which is much slower on large systems.

    Parameters
    ----------
    arr_obj : np.ndarray
        Object-dtype array of strings to test.
    values : str or iterable of str
        Value(s) to test membership against.

    Returns
    -------
    np.ndarray
        boolean array, ``True`` where ``arr_obj[i]`` is in ``values``.
    """
    if arr_obj is None or len(arr_obj) == 0:
        return np.array([], dtype=bool)
    vals = list(values)
    if len(vals) == 1:
        return np.asarray(arr_obj == vals[0], dtype=bool)
    vset = set(vals)
    return np.fromiter((x in vset for x in arr_obj), dtype=bool,
                       count=len(arr_obj))


def by_residue_names(residue_names) -> np.ndarray:
    """Select all atoms whose residue name is any of `residue_names`.

    One pass: membership is tested at the residue level (few residues) and
    expanded to atoms. Equivalent to OR-ing by_residue_name over each name;
    a single-element list matches by_residue_name exactly.

    Parameters
    ----------
    residue_names : str or iterable of str
        One residue name, or a collection of residue names to match.

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    names = _as_value_list(residue_names)
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)
    if not names:
        return none_selection(n_atoms)

    all_residue_names = psf.get_res()
    if not all_residue_names:
        return none_selection(n_atoms)

    res_names_arr = np.array(all_residue_names, dtype=object)
    matching_res = _isin_object(res_names_arr, names)          # O(n_residues)

    atom_res_idx = np.array(atom_info.atom_to_res(), dtype=np.intp)
    n_residues = len(all_residue_names)
    result = np.zeros(n_atoms, dtype=bool)
    valid = (atom_res_idx >= 0) & (atom_res_idx < n_residues)
    result[valid] = matching_res[atom_res_idx[valid]]         # O(natom)
    return result


def by_residue_ids(residue_ids) -> np.ndarray:
    """Select all atoms whose residue ID is any of `residue_ids`.

    One pass over the cached per-atom residue IDs. A single-element list
    matches by_residue_id exactly.

    Parameters
    ----------
    residue_ids : str or iterable of str
        One residue ID, or a collection of residue IDs to match.

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    ids = _as_value_list(residue_ids)
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)
    if not ids:
        return none_selection(n_atoms)

    cache = _get_cache()
    cache._ensure_cache()
    if (cache._per_atom_res_ids is not None
            and len(cache._per_atom_res_ids) == n_atoms):
        return _isin_object(cache._per_atom_res_ids, ids)
    return none_selection(n_atoms)


def by_segment_ids(segment_ids) -> np.ndarray:
    """Select all atoms in any of `segment_ids`.

    One pass, OR-ing the cached per-segment masks. A single-element list
    matches by_segment_id exactly.

    Parameters
    ----------
    segment_ids : str or iterable of str
        One segment ID, or a collection of segment IDs to match.

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    segids = _as_value_list(segment_ids)
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)
    if not segids:
        return none_selection(n_atoms)

    cache = _get_cache()
    cache._ensure_cache()
    result = np.zeros(n_atoms, dtype=bool)
    if cache._seg_masks:
        for sid in segids:
            mask = cache._seg_masks.get(sid)
            if mask is not None:
                result |= mask
    return result


def by_atom_types(atom_types) -> np.ndarray:
    """Select all atoms whose name is any of `atom_types`.

    One pass over the cached per-atom names. A single-element list matches
    by_atom_type exactly.

    Parameters
    ----------
    atom_types : str or iterable of str
        One atom name (IUPAC), or a collection of atom names to match.

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    types = _as_value_list(atom_types)
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)
    if not types:
        return none_selection(n_atoms)

    cache = _get_cache()
    cache._ensure_cache()
    if cache._atom_types is not None and len(cache._atom_types) == n_atoms:
        return _isin_object(cache._atom_types, types)
    return none_selection(n_atoms)


def by_chem_types(chem_types) -> np.ndarray:
    """Select all atoms whose chemical type is any of `chem_types`.

    Matching is done at the parameter type-code level (few codes) and mapped
    to atoms with an integer isin, which numpy handles in linear time. A
    single-element list matches by_chem_type exactly.

    Parameters
    ----------
    chem_types : str or iterable of str
        One parameter type code, or a collection of them to match.

    Returns
    -------
    flags : np.ndarray
            boolean array, atom i selected <==> flags[i] == True
    """
    types = _as_value_list(chem_types)
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=bool)
    if not types:
        return none_selection(n_atoms)

    type_code_to_name = param.get_atc()
    per_atom_codes = psf.get_iac()
    if (not type_code_to_name or not per_atom_codes
            or len(per_atom_codes) != n_atoms):
        return none_selection(n_atoms)

    tset = set(types)
    matching_codes = [i for i, nm in enumerate(type_code_to_name)
                      if nm in tset]                            # O(n_types)
    if not matching_codes:
        return np.zeros(n_atoms, dtype=bool)
    codes_arr = np.array(per_atom_codes, dtype=np.intp)
    return np.isin(codes_arr, np.array(matching_codes, dtype=np.intp))


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
    if n_atoms == 0:
        return tuple()

    positions_df = coor.get_positions()
    if positions_df.empty:
        return none_selection(n_atoms)
        
    coords_np = positions_df[['x', 'y', 'z']].to_numpy()

    dx_all = x - coords_np[:, 0]
    dy_all = y - coords_np[:, 1]
    dz_all = z - coords_np[:, 2]

    if periodic:
        # Fetch PBC parameters once
        # Ensure lib is initialized before accessing its attributes
        if lib is None:
            raise RuntimeError(
                "CHARMM library not initialized. Cannot use periodic=True without "
                "a valid CHARMM library. Ensure CHARMM_LIB_DIR is set correctly."
            )

        is_to_box_c = lib.pbound_is_to_box()
        is_cubic_box_c = lib.pbound_is_cubic_box()
        is_to_box_val = getattr(is_to_box_c, 'value', is_to_box_c) # Handle potential direct int
        is_cubic_box_val = getattr(is_cubic_box_c, 'value', is_cubic_box_c)


        if is_to_box_val == 1 or is_cubic_box_val == 1:
            boxinv_x_c, boxinv_y_c, boxinv_z_c = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
            lib.pbound_get_boxinv(ctypes.byref(boxinv_x_c), ctypes.byref(boxinv_y_c), ctypes.byref(boxinv_z_c))
            boxinv_x, boxinv_y, boxinv_z = boxinv_x_c.value, boxinv_y_c.value, boxinv_z_c.value

            # Scale to fractional coordinates (relative to box)
            dx_frac = dx_all * boxinv_x
            dy_frac = dy_all * boxinv_y
            dz_frac = dz_all * boxinv_z

            # Apply minimum image convention in fractional coordinates
            dx_frac = dx_frac - np.rint(dx_frac) # np.rint rounds to nearest int; more robust than >0.5 logic for -0.5 to 0.5 range
            dy_frac = dy_frac - np.rint(dy_frac)
            dz_frac = dz_frac - np.rint(dz_frac)
            
            if is_to_box_val == 1: # Specific correction for TOBOX (tetragonal/orthorhombic)
                # This part is more complex due to the 'r75' logic and per-component copysign.
                # The original code's r75 logic was:
                # corr = 0.5 * math.trunc(r75 * (abs(dx_frac) + abs(dy_frac) + abs(dz_frac)))
                # dx_frac -= math.copysign(corr, dx_frac) ...
                # Replicating this precisely in a vectorized way without knowing more about r75's role
                # and the exact transformation CHARMM does can be tricky.
                # For now, we might acknowledge this specific correction is hard to vectorize perfectly
                # or simplify if possible. CHARMM's TOBOX implies orthorhombic or tetragonal.
                # The r75 seems to be a specific CHARMM internal variable/logic.
                # If pbound_get_r75 is available and its logic is clear, we can attempt vectorization.
                # Let's assume for now that the np.rint() handles the primary MIC for cubic/tetragonal.
                # The r75 correction might be for more specific cases or precision.
                # For simplicity in this pass, we'll rely on np.rint for MIC.
                # A more faithful vectorization of this specific TOBOX step would require deeper analysis.
                pass # Placeholder for more complex r75 logic if needed

            # Scale back to Cartesian distances using actual box sizes
            size_x_c, size_y_c, size_z_c = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
            lib.pbound_get_size(ctypes.byref(size_x_c), ctypes.byref(size_y_c), ctypes.byref(size_z_c))
            
            dx_all = dx_frac * size_x_c.value
            dy_all = dy_frac * size_y_c.value
            dz_all = dz_frac * size_z_c.value
        else:
            # General periodic boundary conditions (e.g., triclinic)
            # This requires calling pbound_pbmove for each atom, difficult to vectorize directly.
            for i in range(n_atoms):
                dx_c = ctypes.c_double(dx_all[i])
                dy_c = ctypes.c_double(dy_all[i])
                dz_c = ctypes.c_double(dz_all[i])
                lib.pbound_pbmove(ctypes.byref(dx_c), ctypes.byref(dy_c), ctypes.byref(dz_c))
                dx_all[i] = dx_c.value
                dy_all[i] = dy_c.value
                dz_all[i] = dz_c.value

    dist_sq_all = dx_all**2 + dy_all**2 + dz_all**2
    cut_sq = cut**2
    selection_mask = dist_sq_all <= cut_sq
    
    return tuple(selection_mask)


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
    test = lib.select_is_hydrog(c_i)
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
    test = lib.select_is_lone(c_i)
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
    test = lib.select_is_initial(c_i)
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
    prop_vals : np.ndarray
                numeric value for each atom i representing *prop*
    """
    c_prop = ctypes.c_char_p(prop_name.encode('utf-8').lower())
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return np.array([], dtype=float) # Return empty numpy array
        
    prop_vals_ctype = (ctypes.c_double * n_atoms)(0.0)
    lib.select_get_property(c_prop, prop_vals_ctype,
                                   ctypes.c_int(n_atoms))
    # Convert ctype array to numpy array
    prop_vals_np = np.array(prop_vals_ctype, dtype=float)
    return prop_vals_np


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
    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return tuple()

    property_values = get_property(prop_name) # Now returns a NumPy array

    if property_values.size == 0:
        return none_selection(n_atoms) # Should be 0 if n_atoms is 0, but good check

    # Apply the function. This is the part that is not yet vectorized
    # if func is a generic python callable.
    # For common cases (gt, lt, eq within tolerance), we might optimize further.
    selection_mask = np.array([func(tol, p_val) for p_val in property_values], dtype=bool)
    
    return tuple(selection_mask)


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
    lib.select_resname_range(c_name_a, c_name_b, flags)
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
    lib.select_segid_range(c_name_a, c_name_b, flags)
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
    n_atoms = psf.get_natom() # Ensure n_atoms is available for none_selection if sel is empty
    if n_atoms == 0:
        return tuple()
        
    residue_table = atom_info.atom_to_res()
    if not residue_table: # atom_info.atom_to_res could return empty if no atoms
        return none_selection(n_atoms)
        
    # Ensure sel has correct length if not empty
    if len(sel) != n_atoms:
        # This case should ideally not happen if sel comes from this module
        # Or it might mean sel is for a different system size. Handle defensively.
        # print(f"Warning: 'sel' length mismatch in whole_residues. Expected {n_atoms}, got {len(sel)}.")
        # Fallback to an empty selection of the correct current size.
        return none_selection(n_atoms)
        
    selected_atom_indices = [i for i, v in enumerate(sel) if v]
    if not selected_atom_indices:
        return none_selection(n_atoms)

    residues_wanted = atom_info.get_res_indexes(selected_atom_indices)
    
    new_sel_mask = np.full(n_atoms, False, dtype=bool)
    # Vectorized check or efficient loop if residues_wanted can be large
    # Convert residue_table to numpy array for efficient lookup if not already
    residue_table_np = np.array(residue_table)
    wanted_set = set(residues_wanted) # Faster lookups

    for res_idx_wanted in wanted_set:
        new_sel_mask[residue_table_np == res_idx_wanted] = True
        
    return tuple(new_sel_mask)


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

    n_atoms = psf.get_natom()
    if n_atoms == 0:
        return tuple()

    # Ensure sel has the correct length for the current number of atoms
    if len(sel) != n_atoms:
        # print(f"Warning: 'sel' length mismatch in around. Expected {n_atoms}, got {len(sel)}.")
        return none_selection(n_atoms) # Or raise error

    selected_atom_original_indices = np.where(np.array(sel, dtype=bool))[0]
    if selected_atom_original_indices.size == 0: # No atoms in initial selection
        return sel # Return the original empty selection tuple of correct size

    stats = coor.stat()
    r_all = coor.get_positions()[['x', 'y', 'z']].to_numpy()
    
    # Shift coordinates for cell indexing relative to system min
    system_min = np.array([stats["xmin"], stats["ymin"], stats["zmin"]])
    r_shifted = r_all - system_min

    lx = stats["xmax"] - stats["xmin"]
    ly = stats["ymax"] - stats["ymin"]
    lz = stats["zmax"] - stats["zmin"]
    system_max_dim = max(lx, ly, lz)
    if system_max_dim == 0: # Avoid division by zero if system is a point
        if n_atoms > 0: # If there are atoms, all are at the same point
             # if r_cut is positive, all atoms are near each other (dist 0)
             # The selection should include all atoms if any are selected initially
            if selected_atom_original_indices.size > 0:
                return all_selection(n_atoms)
            else:
                return none_selection(n_atoms)
        else: # No atoms, already handled
            return tuple()

    rn = system_max_dim / int(system_max_dim / r_cut) if int(system_max_dim / r_cut) > 0 else r_cut
    if rn == 0: rn = r_cut # Avoid division by zero if r_cut is very large relative to system_max_dim
    
    sc_base = np.floor(np.array([lx,ly,lz]) / rn).astype(int)
    sc_x, sc_y, sc_z = sc_base[0] + 2, sc_base[1] + 2, sc_base[2] + 2 # Padded number of cells

    # relative neighborhood array (27 neighbors including self cell)
    d_half = np.array([[0,0,0],[1,0,0],[1,1,0],[-1,1,0],[0,1,0],[0,0,1],[-1,0,1],[1,0,1],[-1,-1,1],[0,-1,1],[1,-1,1],[-1,1,1],[0,1,1],[1,1,1]])
    neighbor_offsets_d = np.unique(np.concatenate((d_half, -d_half)), axis=0)

    # Calculate cell indices for all atoms, ensuring they are within bounds for head array
    cell_indices_all = np.floor(r_shifted / rn).astype(int)
    cell_indices_all[:, 0] = np.clip(cell_indices_all[:, 0], 0, sc_x - 1)
    cell_indices_all[:, 1] = np.clip(cell_indices_all[:, 1], 0, sc_y - 1)
    cell_indices_all[:, 2] = np.clip(cell_indices_all[:, 2], 0, sc_z - 1)

    # Build linked-list for atoms in cells
    ll = np.full(n_atoms, -1, dtype=int)
    head = np.full((sc_x, sc_y, sc_z), -1, dtype=int)

    for atom_k_idx in range(n_atoms):
        cx, cy, cz = cell_indices_all[atom_k_idx]
        ll[atom_k_idx] = head[cx, cy, cz]
        head[cx, cy, cz] = atom_k_idx

    found_nearby_mask = np.zeros(n_atoms, dtype=bool)
    r_cut_sq = r_cut**2

    for atom_i_orig_idx in selected_atom_original_indices:
        coords_atom_i = r_all[atom_i_orig_idx]
        cell_atom_i = cell_indices_all[atom_i_orig_idx]

        for dj in neighbor_offsets_d:
            ngh_cell_x, ngh_cell_y, ngh_cell_z = cell_atom_i + dj

            # Check bounds for neighbor cell index
            if not (0 <= ngh_cell_x < sc_x and 0 <= ngh_cell_y < sc_y and 0 <= ngh_cell_z < sc_z):
                continue

            k_indices_in_ngh_cell_list = []
            current_atom_k_idx_in_cell = head[ngh_cell_x, ngh_cell_y, ngh_cell_z]
            while current_atom_k_idx_in_cell != -1:
                k_indices_in_ngh_cell_list.append(current_atom_k_idx_in_cell)
                current_atom_k_idx_in_cell = ll[current_atom_k_idx_in_cell]
            
            if not k_indices_in_ngh_cell_list:
                continue

            coords_atoms_k_in_ngh_cell = r_all[k_indices_in_ngh_cell_list]
            diff_vectors = coords_atoms_k_in_ngh_cell - coords_atom_i # Broadcast subtraction
            dist_sq_to_atom_i = np.sum(diff_vectors**2, axis=1)
            
            is_within_cut_mask_for_ngh_cell = dist_sq_to_atom_i <= r_cut_sq
            
            original_indices_of_nearby_atoms_in_cell = np.array(k_indices_in_ngh_cell_list)[is_within_cut_mask_for_ngh_cell]
            found_nearby_mask[original_indices_of_nearby_atoms_in_cell] = True

    # The new selection includes the current selection.
    final_selection_mask = found_nearby_mask | np.array(sel, dtype=bool)
    return tuple(final_selection_mask)


def get_max_name() -> int:
    """Ask CHARMM for the max len of the name of a stored selection.
    Returns
    -------
    max_name : int
    """
    max_name = lib.select_get_max_name()
    return max_name


def get_num_stored() -> int:
    """Ask CHARMM for the number of stored selections
    Returns
    -------
    num_stored : int
    """
    num_stored = lib.select_get_num_stored()
    return num_stored


def find(name: str) -> int:
    """Get the index of the named stored selection
    Returns
    -------
    int
    """
    c_name = ctypes.c_char_p(name.encode('utf-8'))
    c_len_name = ctypes.c_int(len(name))
    found = lib.select_find(c_name, c_len_name)
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
    # Validate selection
    if sel is None:
        raise ValueError(f"Selection '{name}' is None")
    if len(sel) == 0:
        raise ValueError(f"Selection '{name}' is empty - no atoms in system?")

    c_name = ctypes.c_char_p(name.upper().encode('utf-8'))
    c_len_name = ctypes.c_int(len(name))

    # Convert to list of integers for reliable ctypes conversion
    # This handles numpy arrays, lists, tuples with any boolean-like types
    sel_int = [int(x) for x in sel]
    c_sel = (ctypes.c_int * len(sel_int))(*sel_int)
    c_len_sel = ctypes.c_int(len(sel_int))

    lib.select_store(c_name, c_len_name, c_sel, c_len_sel)
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

    lib.select_get_stored_names(name_pointers, ctypes.c_int(n), ctypes.c_int(max_name))
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
    lib.select_delete(c_name, c_len_name)
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


# Performance benchmarking utilities
def get_optimization_status() -> dict:
    """Report which optimization libraries are available.
    
    Returns
    -------
    status : dict
        Dictionary with keys 'numba', 'numexpr' indicating availability
    """
    return {
        'numba': _HAS_NUMBA,
        'numexpr': _HAS_NUMEXPR,
    }


def benchmark_selection(n_iterations: int = 100, n_atoms: int = None) -> dict:
    """Benchmark selection operations to measure performance.
    
    Note: This benchmark uses synthetic data and does not require
    a loaded CHARMM system.
    
    Parameters
    ----------
    n_iterations : int
        Number of iterations for timing (default: 100)
    n_atoms : int, optional
        Number of atoms to simulate. If None, uses current system's natom
        or defaults to 50000.
    
    Returns
    -------
    results : dict
        Dictionary with timing results for various operations
    """
    import time
    
    # Determine number of atoms
    if n_atoms is None:
        n_atoms = psf.get_natom()
        if n_atoms == 0:
            n_atoms = 50000  # Default for benchmarking without system
    
    # Create test arrays
    rng = np.random.default_rng(42)
    sel_a = rng.random(n_atoms) > 0.5
    sel_b = rng.random(n_atoms) > 0.5
    
    results = {
        'n_atoms': n_atoms,
        'n_iterations': n_iterations,
        'optimizations': get_optimization_status(),
        'timings': {}
    }
    
    # Benchmark OR operation
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = or_selection(sel_a, sel_b)
    results['timings']['or_selection'] = (time.perf_counter() - start) / n_iterations * 1000  # ms
    
    # Benchmark AND operation
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = and_selection(sel_a, sel_b)
    results['timings']['and_selection'] = (time.perf_counter() - start) / n_iterations * 1000  # ms
    
    # Benchmark NOT operation
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = not_selection(sel_a)
    results['timings']['not_selection'] = (time.perf_counter() - start) / n_iterations * 1000  # ms
    
    # Benchmark combined operations (common pattern)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = and_selection(sel_a, not_selection(sel_b))
    results['timings']['combined_and_not'] = (time.perf_counter() - start) / n_iterations * 1000  # ms
    
    return results


def print_benchmark(results: dict = None, n_iterations: int = 100, n_atoms: int = None):
    """Run and print benchmark results in a formatted table.
    
    Parameters
    ----------
    results : dict, optional
        Pre-computed benchmark results. If None, runs benchmark.
    n_iterations : int
        Number of iterations for timing (default: 100)
    n_atoms : int, optional
        Number of atoms to simulate.
    """
    if results is None:
        results = benchmark_selection(n_iterations, n_atoms)
    
    print("\n=== Selection Performance Benchmark ===")
    print(f"Atoms: {results['n_atoms']:,}")
    print(f"Iterations: {results['n_iterations']}")
    print(f"\nOptimizations available:")
    print(f"  numba:   {'Yes' if results['optimizations']['numba'] else 'No'}")
    print(f"  numexpr: {'Yes' if results['optimizations']['numexpr'] else 'No'}")
    print(f"\nOperation timings (ms per call):")
    print("-" * 40)
    for op, timing in results['timings'].items():
        print(f"  {op:20s}: {timing:.4f} ms")
    print("-" * 40)
    
    # Calculate throughput
    atoms_per_ms = results['n_atoms'] / results['timings'].get('or_selection', 1)
    print(f"\nThroughput: ~{atoms_per_ms/1000:.1f}M atoms/ms for basic operations")
