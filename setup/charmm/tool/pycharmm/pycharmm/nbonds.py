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
   controlling how energy is calculated to compute forces for
   minimization, dynamics or simply energy commands

Corresponds to CHARMM command `NBONds`

See CHARMM documentation [nbonds](<https://academiccharmm.org/documentation/version/c47b1/nbonds>)
for more information

Functions
=========
- `set_cutnb` -- change cutnb
- `set_ctonnb` -- change ctonnb
- `set_ctofnb` -- change ctofnb
- `set_eps` -- change eps
- `use_cdie` -- turn cdie on
- `use_atom` -- turn atom on
- `use_fswitch` -- turn fswitch on
- `use_vatom` -- turn vatom on
- `use_vfswitch` -- turn vfswitch on
- `configure` -- set nonbonded params from a dict

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
import pycharmm.lib as lib


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
    old_inbfrq = lib.charmm.nbonds_set_inbfrq(ctypes.byref(new_inbfrq))
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
    old_imgfrq = lib.charmm.nbonds_set_imgfrq(ctypes.byref(new_imgfrq))
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

    c_set_cutim = lib.charmm.nbonds_set_cutim
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

    c_set_cutnb = lib.charmm.nbonds_set_cutnb
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

    c_set_ctonnb = lib.charmm.nbonds_set_ctonnb
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

    c_set_ctofnb = lib.charmm.nbonds_set_ctofnb
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

    c_set_eps = lib.charmm.nbonds_set_eps
    c_set_eps.restype = ctypes.c_double

    old_eps = c_set_eps(ctypes.byref(new_eps))
    return old_eps


def use_cdie():
    """Use constant dielectric for radial energy functional form.
    Energy is proportional to 1/R.

    Returns
    -------
    old_cdie : float
               the old cdie
    """
    old_cdie = lib.charmm.nbonds_use_cdie()
    bool(old_cdie)
    return old_cdie


def use_atom():
    """Compute interactions on an atom-atom pair basis

    Returns
    -------
    old_atom : float
               the old atom
    """
    old_atom = lib.charmm.nbonds_use_atom()
    bool(old_atom)
    return old_atom


def use_vatom():
    """Compute the van der waal energy term on an atom-atom pair basis

    Returns
    -------
    old_vatom : float
               the old vatom
    """
    old_vatom = lib.charmm.nbonds_use_vatom()
    bool(old_vatom)
    return old_vatom


def use_fswitch():
    """Use switching function on forces only from CTONNB to CTOFNB

    Returns
    -------
    old_fswitch : float
               the old fswitch
    """
    old_fswitch = lib.charmm.nbonds_use_fswitch()
    bool(old_fswitch)
    return old_fswitch


def use_vfswitch():
    """Use switching function on VDW force from CTONNB to CTOFNB

    Returns
    -------
    old_vfswitch : float
               the old vfswitch
    """
    old_vfswitch = lib.charmm.nbonds_use_vfswitch()
    bool(old_vfswitch)
    return old_vfswitch


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


# Addition Kai Toepfer May 2022
def get_ctonnb():
    """Get ctonnb, distance after which the switching function is active

    Returns
    -------
    old_ctonnb : float
                 the current ctonnb
    """
    
    old_ctonnb = (ctypes.c_double * 1)()
    status = lib.charmm.nbonds_get_ctonnb(old_ctonnb)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem fetching unit cell data.')

    return list(old_ctonnb)[0]


def get_ctofnb():
    """Get ctofnb, distance at which switching function stops being used

    Returns
    -------
    old_ctonnb : float
                 the current ctofnb
    """
    
    old_ctofnb = (ctypes.c_double * 1)()
    status = lib.charmm.nbonds_get_ctofnb(old_ctofnb)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem fetching unit cell data.')

    return list(old_ctofnb)[0]


def update_bnbnd():
    """Update non-bonded exclusion list
    """
    
    lib.charmm.nbonds_update_bnbnd()
    
    return

