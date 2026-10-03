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

"""Evaluation and manipulation of the potential energy of a macromolecular system.

Corresponds to CHARMM command `ENERgy`
See [ENERgy documentation](https://academiccharmm.org/documentation/version/c47b1/energy)

Functions
=========
Display
-------
- `show` -- print energy table to output

Energy Properties (eprop array)
-------------------------------
- `get_total` -- get total potential energy (the table's ENER; see Notes there)
- `get_eprop` -- get energy property by index
- `get_grms` -- get gradient RMS
- `get_num_properties` -- get number of energy properties
- `get_property` -- get property by index
- `get_property_by_name` -- get property by name
- `get_property_names` -- get list of property names
- `get_property_statuses` -- get status flags for properties
- `get_properties` -- get all property values

Energy Terms (eterm array)
--------------------------
- `get_eterm` -- get energy term by index
- `get_bonded` -- get bond energy (BOND)
- `get_angle` -- get angle energy (ANGL)
- `get_urey` -- get Urey-Bradley energy (UREY)
- `get_dihedral` -- get dihedral energy (DIHE)
- `get_improper` -- get improper energy (IMPR)
- `get_vdw` -- get van der Waals energy (VDW)
- `get_elec` -- get electrostatic energy (ELEC)
- `get_num_terms` -- get number of energy terms
- `get_term` -- get term by index
- `get_term_by_name` -- get term by name
- `get_term_names` -- get list of term names
- `get_term_statuses` -- get status flags for terms
- `get_terms` -- get all term values

Combined / Utilities
--------------------
- `get_old_energy` -- get energy from previous update
- `get_energy_change` -- get change in energy between updates
- `get_delta_e` -- alias for get_energy_change
- `get_energy` -- get pandas DataFrame of all active terms
- `from_omm` -- get energy from CHARMM/OpenMM

Constants
=========
Energy term indices (for get_eterm):
    BOND = 1   : Bond stretching energy
    ANGL = 2   : Angle bending energy
    UREY = 3   : Urey-Bradley (1,3) energy
    DIHE = 4   : Dihedral (torsion) energy
    IMPR = 5   : Improper dihedral energy
    VDW  = 6   : van der Waals energy
    ELEC = 7   : Electrostatic energy

OpenMM custom-force buckets (populated when user-added custom forces
are present; see `pycharmm.omm.CUSTOM_FORCE_BUCKETS`):
    CFIN       : Custom internal (BOND/ANGLE/TORSION custom forces)
    CFNB       : Custom nonbonded (NONBONDED/GB)
    CFEX       : Custom external (EXTERNAL)
    CFMB       : Custom many-body (COMPOUND/CENTROID/H_BOND/MANY_PARTICLE)
    CFCV       : Custom collective variables (CV/VOLUME/RMSD/RG)
    NNPO       : Neural-network potential via OpenMM-Torch

MLpot terms (populated once an MLpot model is registered; see
`pycharmm.energy_mlpot` and mlpot.info):
    MLPO = 125 : MLpot internal ML atom potential energy
    MLEL = 126 : MLpot ML-charge/MM-charge electrostatic energy

Energy property indices (for get_eprop). The index name is the one used in
`energym.F90`; the quoted name is what CHARMM prints for it, assigned in
`eutil.F90`:
    EPROP_TOTE  = 1 : 'TOTE' -- total energy (potential + kinetic)
    EPROP_TOTKE = 2 : 'TOTK' -- total kinetic energy
    EPROP_EPOT  = 3 : 'ENER' -- total potential energy
    EPROP_TEMPS = 4 : 'TEMP' -- temperature (from the kinetic energy)
    EPROP_GRMS  = 5 : 'GRMS' -- rms gradient

Note that CHARMM's printed name and its index name disagree for the first
and third properties: the property printed as ENER is at index 3, not 1.

.. deprecated::
    The `PROP_*` constants are deprecated in favour of `EPROP_*`, and warn
    when read. Two of them had their values transposed on exactly the
    confusion above, each holding the index belonging to the other's name:

    - ``PROP_ENER`` (1) -- 1 is TOTE, not ENER. For the total potential
      energy its name promises, use ``EPROP_EPOT`` (3); to keep the present
      value, ``EPROP_TOTE``. Because TOTE is only set by dynamics,
      ``get_eprop(PROP_ENER)`` returns 0.0 after a plain energy evaluation.
    - ``PROP_TOTE`` (3) -- 3 is EPOT, the potential energy, not the total.
      Use ``EPROP_EPOT`` to keep the present value, ``EPROP_TOTE`` for the
      total energy.
    - ``PROP_GRMS`` (5) -- renamed only; use ``EPROP_GRMS``.

    The deprecated constants keep their original values, so no existing
    result changes silently.

Examples
========
>>> import pycharmm.energy as energy

Print out all the energy terms
>>> energy.show()

Get specific energy components
>>> total = energy.get_total()
>>> vdw = energy.get_vdw()
>>> elec = energy.get_elec()
>>> grms = energy.get_grms()

Get energy as a pandas DataFrame
>>> df = energy.get_energy()
>>> print(df)

Get energy term by name
>>> bond_energy = energy.get_term_by_name('BOND')
>>> elec_energy = energy.get_term_by_name('ELEC')

"""

import ctypes
import warnings

import pandas

from pycharmm.loader import lib
import pycharmm.script


# Energy term indices (eterm array, 1-indexed for Fortran)
TERM_BOND = 1  # Bond stretching
TERM_ANGL = 2  # Angle bending
TERM_UREY = 3  # Urey-Bradley 1,3 interaction
TERM_DIHE = 4  # Dihedral (torsion)
TERM_IMPR = 5  # Improper dihedral
TERM_VDW = 6   # van der Waals
TERM_ELEC = 7  # Electrostatic

# MLpot terms. Populated only once an MLpot model has been registered; they
# stay zero (and unnamed, so absent from the energy table and the SKIPE
# listing) otherwise. MLpot is only callable from pyCHARMM, so these are
# exposed here rather than left as magic indices.
TERM_MLPO = 125  # MLpot internal ML atom potential
TERM_MLEL = 126  # MLpot ML-charge/MM-charge electrostatics

# Energy property indices (eprop array, 1-indexed for Fortran).
#
# Named after the index parameters in energym.F90 rather than after the
# four-character names CHARMM prints, because those two disagree in a way
# that has already caused a bug. eutil.F90 assigns:
#
#     CEPROP(TOTE) = 'TOTE'   ! index 1, total energy
#     CEPROP(EPOT) = 'ENER'   ! index 3, total potential energy
#
# so the property printed as ENER is index 3, and index 1 is TOTE. The
# deprecated PROP_ENER/PROP_TOTE constants below were defined the other way
# round: each held the index belonging to the other one's name.
EPROP_TOTE = 1   # 'TOTE' total energy (potential + kinetic); 0.0 until dynamics
EPROP_TOTKE = 2  # 'TOTK' total kinetic energy
EPROP_EPOT = 3   # 'ENER' total potential energy -- the energy table's ENER
EPROP_TEMPS = 4  # 'TEMP' temperature, from the kinetic energy
EPROP_GRMS = 5   # 'GRMS' rms gradient

# Deprecated property constants. Served by the module __getattr__ below so
# that reading one warns; they are deliberately NOT defined here.
_DEPRECATED_PROPS = {
    # name: (value it has always had, replacement, why)
    'PROP_ENER': (
        1, 'EPROP_TOTE',
        "PROP_ENER is 1, but index 1 is TOTE (total energy), not the "
        "property CHARMM prints as ENER. Its documented meaning, total "
        "potential energy, is index 3: use EPROP_EPOT for that, or "
        "EPROP_TOTE to keep the present value. Note that eprop(1) is 0.0 "
        "after a plain energy evaluation, since TOTE is only set by "
        "dynamics -- code reading PROP_ENER for a potential energy has "
        "been silently receiving 0.0",
    ),
    'PROP_TOTE': (
        3, 'EPROP_EPOT',
        "PROP_TOTE is 3, but index 3 is EPOT, the total potential energy, "
        "not the total energy. Use EPROP_EPOT to keep the present value, "
        "or EPROP_TOTE for the total energy its name implies",
    ),
    'PROP_GRMS': (
        5, 'EPROP_GRMS',
        "renamed only; the value is unchanged and was always correct",
    ),
}


def __getattr__(name):
    """Serve the deprecated PROP_* constants with a warning.

    Module-level ``__getattr__`` is only consulted after normal attribute
    lookup fails, so these names must not also be defined above.

    The value returned is the value the constant has always had, even where
    that value is wrong for its name. Silently repointing PROP_ENER from 1
    to 3 would change what existing callers compute without any diagnostic,
    which is worse than a loud warning: a caller that worked around the bug
    by treating PROP_ENER as TOTE would break, and one that hit the bug
    would see its numbers change with no explanation.
    """
    try:
        value, replacement, reason = _DEPRECATED_PROPS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    warnings.warn(
        f"pycharmm.energy.{name} is deprecated; use {replacement}. {reason}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__():
    """Include the deprecated constants in dir(), without warning."""
    return sorted(list(globals()) + list(_DEPRECATED_PROPS))


def show():
    """Print the energy table.
    """
    lib.print_energy()


def get_total():
    """Return the total potential energy, the energy table's ENER.

    Returns
    -------
    float
        eprop(EPROP_EPOT), the total potential energy

    Notes
    -----
    Despite the name, this returns the potential energy, not the total
    energy: eprop index 3, which CHARMM prints as ENER. That is what the
    function has always returned and what its callers depend on, so the
    behaviour is kept and the documentation corrected instead.

    This is also the resolution of a longstanding comment here that read
    "Not sure why this is 3". Index 3 is EPOT, whose printed name is ENER
    (assigned in eutil.F90); index 1 is TOTE, which stays 0.0 until
    dynamics sets it. Reading index 1 after a plain energy evaluation
    therefore yields 0.0, which is why index 3 was needed.

    For the true total energy, potential plus kinetic, after dynamics, use
    ``get_eprop(EPROP_TOTE)``.
    """
    prop_index = ctypes.c_int(EPROP_EPOT)
    lib.get_energy_property.restype = ctypes.c_double
    epot = lib.get_energy_property(ctypes.byref(prop_index))
    return epot


def get_eprop(prop_index):
    """Return the energy property from the eprop array at index prop_index

    Parameters
    ----------
    prop_index : int
        index corresponding to the desired energy term
    Returns
    -------
    float
        eprop(prop_index)
    """
    prop_index = ctypes.c_int(prop_index)
    lib.get_energy_property.restype = ctypes.c_double
    prop = lib.get_energy_property(ctypes.byref(prop_index))
    return prop


def get_grms():
    """Return the current GRMS energy property

    Returns
    -------
    float
        the current GRMS
    """
    prop_index = ctypes.c_int(EPROP_GRMS)
    lib.get_energy_property.restype = ctypes.c_double
    grms = lib.get_energy_property(ctypes.byref(prop_index))
    return grms


def get_eterm(term_index):
    """Return the energy term at index term_index in the eterm array

    Parameters
    ----------
    term_index : int
        index in the eterm array
    Returns
    -------
    float
        the current energy term from eterm(`term_index`)
    """
    term_index = ctypes.c_int(term_index)
    lib.get_energy_term.restype = ctypes.c_double
    term = lib.get_energy_term(ctypes.byref(term_index))
    return term


def get_bonded():
    """Return the current bond energy term

    Returns
    -------
    float
        the current bond energy term
    """
    term_index = ctypes.c_int(1)
    lib.get_energy_term.restype = ctypes.c_double
    bond = lib.get_energy_term(ctypes.byref(term_index))
    return bond


def get_angle():
    """Return the current angle energy term

    Returns
    -------
    float
        the current angle energy term
    """
    term_index = ctypes.c_int(2)
    lib.get_energy_term.restype = ctypes.c_double
    angle = lib.get_energy_term(ctypes.byref(term_index))
    return angle


def get_urey():
    """Return the current Urey-Bradley energy term

    Returns
    -------
    float
        Urey-Bradley energy term
    """
    term_index = ctypes.c_int(3)
    lib.get_energy_term.restype = ctypes.c_double
    urey = lib.get_energy_term(ctypes.byref(term_index))
    return urey


def get_dihedral():
    """Return the current dihedral energy term

    Returns
    -------
    float
        Dihedral energy term
    """
    term_index = ctypes.c_int(4)
    lib.get_energy_term.restype = ctypes.c_double
    dihe = lib.get_energy_term(ctypes.byref(term_index))
    return dihe


def get_improper():
    """Return the current improper energy term

    Returns
    -------
    float
        Improper energy term
    """
    term_index = ctypes.c_int(5)
    lib.get_energy_term.restype = ctypes.c_double
    impr = lib.get_energy_term(ctypes.byref(term_index))
    return impr


def get_vdw():
    """Return the current van der Waals energy term

    Returns
    -------
    float
        the current van der Waals energy term
    """
    term_index = ctypes.c_int(6)
    lib.get_energy_term.restype = ctypes.c_double
    vdw = lib.get_energy_term(ctypes.byref(term_index))
    return vdw


def get_elec():
    """Return the current electrostatic energy term

    Returns
    -------
    float
        the current electrostatic energy term
    """
    term_index = ctypes.c_int(7)
    lib.get_energy_term.restype = ctypes.c_double
    elec = lib.get_energy_term(ctypes.byref(term_index))
    return elec


def get_num_properties():
    """Return the current number of properties

    This count includes non-active properties according to qeprop
    (len eprop array)

    Returns
    -------
    int
        total number of properties
    """
    num_eprops = lib.get_num_eprops()
    return num_eprops


def get_property(index):
    """Return the energy property from the `eprop` array at index

    Reads eprop(index) from the CHARMM shared library

    Parameters
    ----------
    index : int
        index of property according to CHARMM
    Returns
    -------
    float
        the value of the property stored in CHARMM
    """
    index = ctypes.c_int(index)
    lib.get_energy_property.restype = ctypes.c_double
    prop = lib.get_energy_property(ctypes.byref(index))
    return prop


def get_property_name_size():
    """Return the character count for each property name in CHARMM

    This count is fixed in CHARMM.
    Does not include the C string termination char

    Returns
    -------
    int
        length of the property name
    """
    name_size = lib.get_eprop_name_size()
    return name_size


def _get_property_name_array():
    """Return CEPROP[1..LENENP] as a list of stripped strings, including
    empty entries for unused slots.

    Length matches the Fortran array, so position i corresponds to
    Fortran index i+1.  Used by lookup-by-name and DataFrame builders
    where positional alignment with EPROP/QEPROP matters.
    """
    num_eprops = get_num_properties()
    name_size = get_property_name_size()
    name_buffers = [ctypes.create_string_buffer(name_size) for _ in range(num_eprops)]
    name_pointers = (ctypes.c_char_p * num_eprops)(*map(ctypes.addressof, name_buffers))
    lib.get_eprop_names(name_pointers)
    return [b.value.decode(errors="ignore").strip() for b in name_buffers]


def get_property_names():
    """Get a list of all energy properties

    This list of names includes non-active properties according to qeprop.
    This list is just the ceprop array

    Returns
    -------
    list[str]
        a list of all energy properties stored in CHARMM
    """
    return [n for n in _get_property_name_array() if n]


def get_property_by_name(name):
    """Return the value of the named energy property

    Returns `eprop(index)` where index satisfies `ceprop(index) == name`

    Parameters
    ----------
    name : str
        name of desired property
    Returns
    -------
    float
        value of named property
    """
    target = name.upper()
    # Walk the full CEPROP array (gaps included) so the resulting
    # Fortran index aligns with EPROP.  Stripping empties first (as
    # get_property_names does for display) breaks the index for any
    # property past a gap.
    for i, n in enumerate(_get_property_name_array()):
        if n == target:
            return get_property(i + 1)  # Fortran array indexing starts at 1
    raise ValueError(f"unknown energy property name: {name!r}")


def get_property_statuses():
    """Get a list of the status for each energy property.

    This list includes non-active properties.
    This list is just the qeprop array.

    Returns
    -------
    list[bool]
        a list of the status for each energy property stored in CHARMM
    """
    num_eprops = get_num_properties()
    statuses = (ctypes.c_int * num_eprops)()
    lib.get_eprop_statuses(statuses)
    statuses = [bool(i) for i in statuses]
    return statuses


def get_properties():
    """Get a list of the value for each energy property

    This list includes non-active properties according to qeprop.
    This list is just the eprop array.

    Returns
    -------
    list[float]
        a list of the value for each energy property stored in CHARMM
    """
    num_eprops = get_num_properties()
    props = (ctypes.c_double * num_eprops)()
    lib.get_energy_properties(props)
    props = props[0:num_eprops]
    return props


def get_num_terms():
    """Return the current number of terms

    This count includes non-active terms according to qeterm
    (len eterm array)

    Returns
    -------
    int
        total number of terms
    """
    for func_name in ("get_num_terms", "get_num_eterms"):
        try:
            return getattr(lib, func_name)()
        except AttributeError:
            continue
    raise AttributeError(
        "CHARMM shared library is missing both get_num_terms and get_num_eterms"
    )


def get_term(index):
    """Return the energy term value at index in the CHARMM shared library

    Parameters
    ----------
    index : int
        index of the term in CHARMM

    Returns
    -------
    float
        the current energy term from eterm(index)
    """
    index = ctypes.c_int(index)
    lib.get_energy_term.restype = ctypes.c_double
    term = lib.get_energy_term(ctypes.byref(index))
    return term


def get_term_name_size():
    """Return the num of chars of each energy term name from CHARMM

    Returns the fixed size of each elt of the ceterm array.
    Does not include the C string termination character

    Returns
    -------
    int
        the size of the name for each energy term
    """
    for func_name in ("get_term_name_size", "get_eterm_name_size"):
        try:
            return getattr(lib, func_name)()
        except AttributeError:
            continue
    raise AttributeError(
        "CHARMM shared library is missing both get_term_name_size and get_eterm_name_size"
    )


def _get_term_name_array():
    """Return CETERM[1..LENENT] as a list of stripped strings, including
    empty entries for unused slots.

    Length matches the Fortran array, so position i corresponds to
    Fortran index i+1.  Used by lookup-by-name and DataFrame builders
    where positional alignment with ETERM/QETERM matters; stripping
    empties (as get_term_names does for display) silently misindexes
    any term past a gap, e.g. NNPO (114) and the CF* buckets (115-119)
    sit past several KEY_* gaps in energym.F90.
    """
    num_terms = get_num_terms()
    name_size = get_term_name_size()
    name_buffers = [ctypes.create_string_buffer(name_size) for _ in range(num_terms)]
    name_pointers = (ctypes.c_char_p * num_terms)(*map(ctypes.addressof, name_buffers))
    for func_name in ("get_term_names", "get_eterm_names"):
        try:
            getattr(lib, func_name)(name_pointers)
            break
        except AttributeError:
            continue
    else:
        raise AttributeError(
            "CHARMM shared library is missing both get_term_names and get_eterm_names"
        )
    return [b.value.decode(errors="ignore").strip() for b in name_buffers]


def get_term_names():
    """Get a list of all energy term names

    This list of names includes non-active terms according to qeterm.
    This list is just the ceterm array

    Returns
    -------
    list[str]
        a list of all energy term names stored in CHARMM
    """
    return [n for n in _get_term_name_array() if n]


def get_term_by_name(name):
    """Return the named energy term from the eterm array

    Returns
    -------
    float
        Value of the named energy term from the `eterm` array
    """
    target = name.upper()
    for i, n in enumerate(_get_term_name_array()):
        if n == target:
            return get_term(i + 1)  # Fortran array indexing starts at 1
    raise ValueError(f"unknown energy term name: {name!r}")


def get_term_statuses():
    """Get a list of the status for each energy term

    This list includes non-active terms.
    This list is just the qeterm array

    Returns
    -------
    list[bool]
        a list of the status for each energy term stored in CHARMM
    """
    num_terms = get_num_terms()
    statuses = (ctypes.c_int * num_terms)()
    for func_name in ("get_term_statuses", "get_eterm_statuses"):
        try:
            getattr(lib, func_name)(statuses)
            break
        except AttributeError:
            continue
    else:
        raise AttributeError(
            "CHARMM shared library is missing both get_term_statuses and get_eterm_statuses"
        )
    statuses = [bool(i) for i in statuses]
    return statuses


def get_terms():
    """Get a list of the value for each energy term

    This list includes non-active terms according to qeprop.
    This list is just the eterm array

    Returns
    -------
    list[float]
        a list of the value for each energy term stored in CHARMM
    """
    num_terms = get_num_terms()
    props = (ctypes.c_double * num_terms)()
    lib.get_energy_terms(props)
    props = props[0:num_terms]
    return props


def get_old_energy():
    """Return the ENER term from the previous energy update.

    This may differ substantially from the current value of ENER.

    Returns
    -------
    float
        the *ENER* term from the previous energy update
    """
    lib.get_old_energy.restype = ctypes.c_double
    old_energy = lib.get_old_energy()
    return old_energy


def get_energy_change():
    """Returns the change in ENER (float) from the penultimate and the last energy update.

    Returns
    -------
    float
        `get_old_energy() - get_property_by_name('ENER')`
    """
    old_energy = get_old_energy()
    new_energy = get_property_by_name('ENER')
    return old_energy - new_energy


def get_delta_e():
    """Return the current Delta-E energy property.
    An alias for `get_energy_change()`
    """
    return get_energy_change()


def get_energy(**kwargs):
    """Get a Pandas dataframe of *ENER*, *GRMS*, and all active eterms

    Parameters
    ----------
    **kwargs: dict
        extra settings to pass to the CHARMM energy command,
        e.g. ``omm=True`` to evaluate via OpenMM

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe with columns *ENER*, *GRMS*, and eterm names
    """
    if kwargs:
        energy_cmd = pycharmm.script.CommandScript('energy', **kwargs)
        energy_cmd.run()
    else:
        show()  # make sure energy props and terms are up to date

    prop_names = ['ENER', 'GRMS']
    props = [get_property_by_name(name) for name in prop_names]

    prop_names = prop_names + ['DELTA']
    props = props + [get_energy_change()]

    term_statuses = get_term_statuses()
    # Use the unfiltered CETERM array so positions align with QETERM/ETERM
    # (length LENENT).  get_term_names() strips empties, which would
    # silently misalign columns past any gap in CETERM.
    term_names = _get_term_name_array()
    terms = get_terms()

    active_term_names = list()
    active_terms = list()
    for status, name, term in zip(term_statuses, term_names, terms):
        if status and name:
            active_term_names.append(name)
            active_terms.append(term)

    energy_cols = prop_names + active_term_names
    energy_row = props + active_terms

    energy = pandas.DataFrame(columns=energy_cols)
    energy.loc[len(energy.index)] = energy_row
    return energy

def from_omm(**kwargs):
    """get the energy from charmm/openmm

    Parameters
    ----------
    **kwargs: dict
        extra settings to pass to the CHARMM command
    """
    omm_energy = pycharmm.script.CommandScript('energy',
                                               omm=True,
                                               **kwargs)
    omm_energy.run()
