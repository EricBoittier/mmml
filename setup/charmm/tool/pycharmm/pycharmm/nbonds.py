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

"""Functions to configure nonbonded interactions

IMPORTANT: PME/EWALD electrostatics require orthorhombic crystal boxes!
Non-orthorhombic crystals (monoclinic, triclinic, hexagonal, rhombohedral,
truncated octahedron, rhombic dodecahedron) will cause CHARMM to die with:
    "The periodic box is non orthorhombic (EWALD wont work)"

Use check_pme_crystal_compatibility() to verify before enabling PME.
   controlling how energy is calculated to compute forces for
   minimization, dynamics or simply energy commands

Corresponds to CHARMM command `NBONds`

See CHARMM documentation [nbonds](<https://academiccharmm.org/documentation/version/c47b1/nbonds>)
for more information

Functions
=========
Setters
-------
- `set_cutnb` -- change cutnb (nonbond list cutoff distance)
- `set_ctonnb` -- change ctonnb (switching function start distance)
- `set_ctofnb` -- change ctofnb (switching function end distance)
- `set_cutim` -- change cutim (image update cutoff distance)
- `set_eps` -- change eps (dielectric constant)
- `set_inbfrq` -- change inbfrq (nonbond list update frequency)
- `set_imgfrq` -- change imgfrq (image list update frequency)

Getters
-------
- `get_cutnb` -- get current cutnb value
- `get_ctonnb` -- get current ctonnb value
- `get_ctofnb` -- get current ctofnb value

Toggles
-------
- `use_cdie` -- use constant dielectric (1/R energy form)
- `use_atom` -- compute interactions on atom-atom pair basis
- `use_fswitch` -- use switching function on forces only
- `use_vatom` -- compute VDW on atom-atom pair basis
- `use_vfswitch` -- use switching function on VDW forces

Utilities
---------
- `configure` -- set nonbonded params from a dict
- `update_bnbnd` -- update non-bonded exclusion list
- `update_nbxmod` -- update NBXMod parameter

Examples
========
Setup nonbonded interactions without PME
>>> import pycharmm
>>> import pycharmm.nbonds as nbonds
>>> nbonds.configure(
...   cutnb=18.0,
...   ctonnb=15.0,
...   ctofnb=13.0,
...   eps=1.0,
...   cdie=True,
...   atom=True,
...   fswitch=True,
...   vatom=True,
...   vfswitch=True)

Another way to setup the nonbonded interactions
>>> nb_dict = {'cutnb': 18.0, 'ctonnb': 15.0, 'ctofnb': 13.0, 'eps': 1.0, 'cdie': True, 'atom': True, 'fswitch': True, 'vatom': True, 'vfswitch': True}
>>> my_nbonds=pycharmm.NonBondedScript(**nb_dict)
>>> my_nbonds.run()
"""

import ctypes
from pycharmm.loader import lib


def set_inbfrq(new_inbfrq):
    """Change inbfrq, the update frequency for the nonbonded list

    Update frequency for the nonbonded list. Used in the subroutine ENERGY()
    to decide whether to update the nonbond list. When set to :

     0 --> no updates of the list will be done.

    +n --> an update is done every time  MOD(ECALLS,n).EQ.0  . This is the old
           frequency scheme, where an update is done every n steps of dynamics
           or minimization.

    -1 --> heuristic testing is performed every time ENERGY() is called and
           a list update is done if necessary. This is the default, because
           it is both safer and more economical than frequency-updating.

    Parameters
    ----------
    new_inbfrq : integer
                 the new update frequency for the nonbonded list

    Returns
    -------
    old_inbfrq : integer
                 the old inbfrq
    """
    new_inbfrq = ctypes.c_int(new_inbfrq)
    old_inbfrq = lib.nbonds_set_inbfrq(ctypes.byref(new_inbfrq))
    return old_inbfrq


def set_imgfrq(new_imgfrq):
    """Change imgfrq, the update frequency for the image list

    Update frequency for the image list. Used in the subroutine ENERGY()
    to decide whether to update the image list. When set to :

     0 --> no updates of the list will be done.

    +n --> an update is done every time  MOD(ECALLS,n).EQ.0  . This is the old
           frequency scheme, where an update is done every n steps of dynamics
           or minimization.

    -1 --> heuristic testing is performed every time ENERGY() is called and
           a list update is done if necessary. This is the default, because
           it is both safer and more economical than frequency-updating.


    Parameters
    ----------
    new_imgfrq : integer
                 the new update frequency for the image list

    Returns
    -------
    old_imgfrq : integer
                 the old imgfrq
    """
    new_imgfrq = ctypes.c_int(new_imgfrq)
    old_imgfrq = lib.nbonds_set_imgfrq(ctypes.byref(new_imgfrq))
    return old_imgfrq


def set_cutim(new_cutim):
    """Change cutim, image update cutoff distance

    Parameters
    ----------
    new_cutim : float
                the new cutim

    Returns
    -------
    old_cutim : float
                the old cutim
    """
    new_cutim = ctypes.c_double(new_cutim)

    c_set_cutim = lib.nbonds_set_cutim
    c_set_cutim.restype = ctypes.c_double

    old_cutim = c_set_cutim(ctypes.byref(new_cutim))
    return old_cutim


def set_cutnb(new_cutnb):
    """Change cutnb, the distance cutoff for interacting particle pairs

    Parameters
    ----------
    new_cutnb : float
                the new cutnb

    Returns
    -------
    old_cutnb : float
                the old cutnb
    """
    new_cutnb = ctypes.c_double(new_cutnb)

    c_set_cutnb = lib.nbonds_set_cutnb
    c_set_cutnb.restype = ctypes.c_double

    old_cutnb = c_set_cutnb(ctypes.byref(new_cutnb))
    return old_cutnb


def set_ctonnb(new_ctonnb):
    """Change ctonnb, distance after which the switching function is active

    Parameters
    ----------
    new_ctonnb : float
                 the new ctonnb

    Returns
    -------
    old_ctonnb : float
                 the old ctonnb
    """
    new_ctonnb = ctypes.c_double(new_ctonnb)

    c_set_ctonnb = lib.nbonds_set_ctonnb
    c_set_ctonnb.restype = ctypes.c_double

    old_ctonnb = c_set_ctonnb(ctypes.byref(new_ctonnb))
    return old_ctonnb


def set_ctofnb(new_ctofnb):
    """Change ctofnb, distance at which switching function stops being used

    Parameters
    ----------
    new_ctofnb : float
                 the new ctofnb

    Returns
    -------
    old_ctofnb : float
                 the old ctofnb
    """
    new_ctofnb = ctypes.c_double(new_ctofnb)

    c_set_ctofnb = lib.nbonds_set_ctofnb
    c_set_ctofnb.restype = ctypes.c_double

    old_ctofnb = c_set_ctofnb(ctypes.byref(new_ctofnb))
    return old_ctofnb


def set_eps(new_eps):
    """Change eps, the dielectric constant for extened electrostatics routines

    Parameters
    ----------
    new_eps : float
              the new eps

    Returns
    -------
    old_eps : float
              the old eps
    """
    new_eps = ctypes.c_double(new_eps)

    c_set_eps = lib.nbonds_set_eps
    c_set_eps.restype = ctypes.c_double

    old_eps = c_set_eps(ctypes.byref(new_eps))
    return old_eps


def use_cdie():
    """Use constant dielectric for radial energy functional form.
    Energy is proportional to 1/R.

    Returns
    -------
    old_cdie : bool
        the previous cdie setting
    """
    old_cdie = lib.nbonds_use_cdie()
    return bool(old_cdie)


def use_atom():
    """Compute interactions on an atom-atom pair basis

    Returns
    -------
    old_atom : bool
        the previous atom setting
    """
    old_atom = lib.nbonds_use_atom()
    return bool(old_atom)


def use_vatom():
    """Compute the van der waal energy term on an atom-atom pair basis

    Returns
    -------
    old_vatom : bool
        the previous vatom setting
    """
    old_vatom = lib.nbonds_use_vatom()
    return bool(old_vatom)


def use_fswitch():
    """Use switching function on forces only from CTONNB to CTOFNB

    Returns
    -------
    old_fswitch : bool
        the previous fswitch setting
    """
    old_fswitch = lib.nbonds_use_fswitch()
    return bool(old_fswitch)


def use_vfswitch():
    """Use switching function on VDW force from CTONNB to CTOFNB

    Returns
    -------
    old_vfswitch : bool
        the previous vfswitch setting
    """
    old_vfswitch = lib.nbonds_use_vfswitch()
    return bool(old_vfswitch)


def configure(**kwargs):
    """Set nonbonded parameters from a dictionary of names and values

    Parameters
    ----------
    **kwargs: dict
        a dictionary of parameter names and their desired values

    Returns
    -------
    bool
        True if everything went well
    """
    glob = globals()
    set_pairs = list()
    uses = list()
    for k, v in kwargs.items():
        setter = glob.get('set_' + k, None)
        toggle = glob.get('use_' + k, None)
        if setter:
            set_pairs.append((setter, v))
        elif toggle:
            uses.append(toggle)
        else:
            raise NameError('function for ' + str(k) + ' not found')

    for f, a in set_pairs:
        f(a)

    for f in uses:
        f()

    return True


def update_bnbnd():
    """Update non-bonded exclusion list
    """
    
    lib.nbonds_update_bnbnd()
    
    return


def _nbond_getters_return_double() -> bool:
    """c52a1 getters return ``real(c_double)``. Older libcharmm writes an out-argument and returns status.

    ``blockdata_is_active`` arrived in the same API bump and is absent from the older library.
    """
    try:
        lib.blockdata_is_active
    except AttributeError:
        return False
    return True


def _get_nbond_distance(symbol: str) -> float:
    getter = getattr(lib, symbol)
    if _nbond_getters_return_double():
        getter.restype = ctypes.c_double
        getter.argtypes = []
        return float(getter())
    slot = (ctypes.c_double * 1)()
    getter.restype = ctypes.c_int
    getter.argtypes = [ctypes.POINTER(ctypes.c_double)]
    if not bool(getter(slot)):
        raise RuntimeError(f"There was a problem fetching {symbol}.")
    return float(slot[0])


def get_cutnb():
    """Get the current value of cutnb from CHARMM

    Returns
    -------
    float
        the current cutnb value
    """
    return _get_nbond_distance("nbonds_get_cutnb")


def get_ctonnb():
    """Get the current value of ctonnb from CHARMM

    Returns
    -------
    float
        the current ctonnb value
    """
    return _get_nbond_distance("nbonds_get_ctonnb")


def get_ctofnb():
    """Get the current value of ctofnb from CHARMM

    Returns
    -------
    float
        the current ctofnb value
    """
    return _get_nbond_distance("nbonds_get_ctofnb")


def update_nbxmod():
    """Update the NBXMod parameter in CHARMM based on current nonbond settings.
    """
    lib.nbonds_update_nbxmod()


def get_primary_pair_count():
    """Return the number of primary-cell pairs in the current JNB list.

    Requires CHARMM build exporting ``nbonds_get_primary_pair_count``.
    Returns ``None`` when unavailable.
    """
    try:
        getter = lib.nbonds_get_primary_pair_count
    except AttributeError:
        return None
    return int(getter())


def export_primary_pairs(*, max_pairs: int | None = None):
    """Export primary-cell JNB pairs as 0-based ``(i, j)`` with ``i < j``.

    Returns ``None`` when the C API is unavailable.
    """
    try:
        exporter = lib.nbonds_export_primary_pairs
        counter = lib.nbonds_get_primary_pair_count
    except AttributeError:
        return None
    cap = int(max_pairs) if max_pairs is not None else int(counter())
    if cap <= 0:
        return [], []
    out_i = (ctypes.c_int * cap)()
    out_j = (ctypes.c_int * cap)()
    out_count = (ctypes.c_int * 1)()
    status = exporter(out_i, out_j, ctypes.c_int(cap), out_count)
    if not bool(status):
        raise RuntimeError("nbonds_export_primary_pairs failed")
    n = int(out_count[0])
    return [int(out_i[k]) for k in range(n)], [int(out_j[k]) for k in range(n)]
