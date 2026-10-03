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

"""Query the residue topology file (RTF) currently loaded in CHARMM

After reading an RTF (see `pycharmm.read.rtf`), these functions report
which residues and atom types it defines, so a script can check whether
the topology it loaded contains what it needs.

Functions
=========
- `get_residue_names` -- get the list of residue names in the loaded RTF
- `get_num_residues` -- get the number of residues in the loaded RTF
- `get_atom_type_names` -- get the list of atom-type names in the loaded RTF
- `get_num_atom_types` -- get the number of atom types in the loaded RTF
- `get_rtf_info` -- get residue and atom-type names as a single dict
"""

import ctypes

from pycharmm.loader import lib
from pycharmm.trace import traced


def _get_names(num_fn, fill_fn):
    """Get the non-blank names from a (count, fill-buffers) C-API pair

    The C arrays are sized by their table length, which can exceed the
    number of names actually defined: atom-type codes in particular are
    often sparse (e.g. the all22 protein RTF assigns codes 1..99 with
    gaps), leaving blank slots that are dropped here.

    Parameters
    ----------
    num_fn : callable
             C-API function returning the table length
    fill_fn : callable
              C-API function that fills an array of c_char_p name buffers

    Returns
    -------
    names : string list
            the non-blank names from the table
    """
    n = int(num_fn())
    if n <= 0:
        return []

    width = int(lib.rtf_name_max())
    # +1 so there is always a trailing NUL: f2c_string copies the fixed
    # width without terminating, and create_string_buffer zero-fills.
    buffers = [ctypes.create_string_buffer(width + 1) for _ in range(n)]
    pointers = (ctypes.c_char_p * n)(*map(ctypes.addressof, buffers))

    fill_fn(pointers)

    names = (b.value.decode(errors='ignore').strip() for b in buffers)
    return [name for name in names if name]


@traced('rtf.get_residue_names')
def get_residue_names():
    """Get a list of the residue names in the currently-loaded RTF

    Returns
    -------
    names : string list
            residue names, e.g. ['ALA', 'ARG', 'ASN', ...]
    """
    return _get_names(lib.rtf_num_residues, lib.rtf_residue_names)


def get_num_residues():
    """Get the number of residues defined in the currently-loaded RTF

    Returns
    -------
    n : integer
        number of residues defined in the RTF
    """
    return len(get_residue_names())


@traced('rtf.get_atom_type_names')
def get_atom_type_names():
    """Get a list of the atom-type names in the currently-loaded RTF

    Returns
    -------
    names : string list
            atom-type names, e.g. ['H', 'HC', 'NH1', 'CT1', ...]
    """
    return _get_names(lib.rtf_num_atom_types, lib.rtf_atom_type_names)


def get_num_atom_types():
    """Get the number of distinct atom types in the currently-loaded RTF

    Counts the named atom types, not the size of the (possibly sparse)
    atom-type-code table.

    Returns
    -------
    n : integer
        number of distinct atom types defined in the RTF
    """
    return len(get_atom_type_names())


def get_rtf_info():
    """Get the residue and atom-type names of the currently-loaded RTF

    Returns
    -------
    info : dict
           a dict with keys 'residues' and 'atom_types', each mapping to
           the corresponding list of names
    """
    return {
        'residues': get_residue_names(),
        'atom_types': get_atom_type_names(),
    }
