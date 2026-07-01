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

Examples
========
>>> import pycharmm.energy as energy

Print out all the energy terms  
>>> energy.show()

"""

import ctypes

import pandas

import pycharmm.lib as lib


# TODO:
# finish module docu


def show():
    """Print the energy table.
    """
    lib.charmm.print_energy()


def get_total():
    """Return the current TOTE energy property (total energy)

    Returns
    ------- 
    tote : float  
        the current TOTE
    """
    # Not sure why this is 3, but needed to change to 3 to get
    # total energy
    prop_index = ctypes.c_int(3)
    lib.charmm.get_energy_property.restype = ctypes.c_double
    tote = lib.charmm.get_energy_property(ctypes.byref(prop_index))
    return tote


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
    lib.charmm.get_energy_property.restype = ctypes.c_double
    prop = lib.charmm.get_energy_property(ctypes.byref(prop_index))
    return prop


def get_grms():
    """Return the current GRMS energy property

    Returns
    -------
    float
        the current GRMS
    """
    prop_index = ctypes.c_int(5)
    lib.charmm.get_energy_property.restype = ctypes.c_double
    grms = lib.charmm.get_energy_property(ctypes.byref(prop_index))
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
    lib.charmm.get_energy_term.restype = ctypes.c_double
    term = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return term


def get_bonded():
    """Return the current bond energy term 

    Returns
    -------
    float
        the current bond energy term
    """
    term_index = ctypes.c_int(1)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    bond = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return bond


def get_angle():
    """Return the current angle energy term 

    Returns
    -------
    float
        the current angle energy term 
    """
    term_index = ctypes.c_int(2)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    angle = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return angle


def get_urey():
    """Return the current Urey-Bradley energy term 

    Returns
    -------
    float
        Urey-Bradley energy term 
    """
    term_index = ctypes.c_int(3)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    urey = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return urey


def get_dihedral():
    """Return the current dihedral energy term

    Returns
    -------
    float
        Dihedral energy term 
    """
    term_index = ctypes.c_int(4)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    dihe = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return dihe


def get_improper():
    """Return the current improper energy term 

    Returns
    -------
    float
        Improper energy term 
    """
    term_index = ctypes.c_int(5)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    impr = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return impr


def get_vdw():
    """Return the current van der Waals energy term 

    Returns
    -------
    float
        the current van der Waals energy term
    """
    term_index = ctypes.c_int(6)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    vdw = lib.charmm.get_energy_term(ctypes.byref(term_index))
    return vdw


def get_elec():
    """Return the current electrostatic energy term 

    Returns
    -------
    float
        the current electrostatic energy term 
    """
    term_index = ctypes.c_int(7)
    lib.charmm.get_energy_term.restype = ctypes.c_double
    elec = lib.charmm.get_energy_term(ctypes.byref(term_index))
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
    num_eprops = lib.charmm.get_num_eprops()
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
    lib.charmm.get_energy_property.restype = ctypes.c_double
    prop = lib.charmm.get_energy_property(ctypes.byref(index))
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
    eprop_name_size = lib.charmm.get_eprop_name_size()
    return eprop_name_size


def get_property_names():
    """Get a list of all energy properties

    This list of names includes non-active properties according to qeprop.  
    This list is just the ceprop array

    Returns
    -------
    list[str]
        a list of all energy properties stored in CHARMM
    """
    num_eprops = get_num_properties()
    eprop_name_size = get_property_name_size()
    labels_bufs = [ctypes.create_string_buffer(eprop_name_size) for _ in range(num_eprops)]
    labels_ptrs = (ctypes.c_char_p * num_eprops)(*map(ctypes.addressof, labels_bufs))

    lib.charmm.get_eprop_names(labels_ptrs)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:num_eprops]]
    labels = [label.strip() for label in labels_str]
    labels = [label for label in labels if label]
    return labels


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
    names = get_property_names()
    index = names.index(name.upper())
    prop = get_property(index + 1)  # fortran array indexing starts at 1
    return prop


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
    lib.charmm.get_eprop_statuses(statuses)
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
    lib.charmm.get_energy_properties(props)
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
    num_eterms = lib.charmm.get_num_eterms()
    return num_eterms


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
    lib.charmm.get_energy_term.restype = ctypes.c_double
    term = lib.charmm.get_energy_term(ctypes.byref(index))
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
    eterm_name_size = lib.charmm.get_eterm_name_size()
    return eterm_name_size


def get_term_names():
    """Get a list of all energy term names

    This list of names includes non-active terms according to qeterm.  
    This list is just the ceterm array

    Returns
    -------
    list[str]
        a list of all energy term names stored in CHARMM
    """
    num_eterms = get_num_terms()
    eterm_name_size = get_term_name_size()
    labels_bufs = [ctypes.create_string_buffer(eterm_name_size) for _ in range(num_eterms)]
    labels_ptrs = (ctypes.c_char_p * num_eterms)(*map(ctypes.addressof, labels_bufs))

    lib.charmm.get_eterm_names(labels_ptrs)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:num_eterms]]
    labels = [label.strip() for label in labels_str]
    labels = [label for label in labels if label]
    return labels


def get_term_by_name(name):
    """Return the named energy term from the eterm array

    Returns
    -------
    float 
        Value of the named energy term from the `eterm` array
    """
    names = get_term_names()
    index = names.index(name.upper())
    term = get_term(index + 1)  # fortran array indexing starts at 1
    return term


def get_term_statuses():
    """Get a list of the status for each energy term

    This list includes non-active terms.  
    This list is just the qeterm array

    Returns
    -------
    list[bool]
        a list of the status for each energy term stored in CHARMM
    """
    num_eterms = get_num_terms()
    statuses = (ctypes.c_int * num_eterms)()
    lib.charmm.get_eterm_statuses(statuses)
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
    num_eterms = get_num_terms()
    props = (ctypes.c_double * num_eterms)()
    lib.charmm.get_energy_terms(props)
    props = props[0:num_eterms]
    return props


def get_old_energy():
    """Return the ENER term from the previous energy update.  

    This may differ substantially from the current value of ENER.  

    Returns
    -------
    float
        the *ENER* term from the previous energy update
    """
    lib.charmm.get_old_energy.restype = ctypes.c_double    
    old_energy = lib.charmm.get_old_energy()
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


def get_energy():
    """Get a Pandas dataframe of *ENER*, *GRMS*, and all active eterms

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe with columns *ENER*, *GRMS*, and eterm names
    """
    show()  # make sure energy props and terms are up to date

    prop_names = ['ENER', 'GRMS']
    props = [get_property_by_name(name) for name in prop_names]

    prop_names = prop_names + ['DELTA']
    props = props + [get_energy_change()]
    
    term_statuses = get_term_statuses()
    term_names = get_term_names()
    terms = get_terms()

    active_term_names = list()
    active_terms = list()
    for status, name, term in zip(term_statuses, term_names, terms):
        if status:
            active_term_names.append(name)
            active_terms.append(term)

    energy_cols = prop_names + active_term_names
    energy_row = props + active_terms

    energy = pandas.DataFrame(columns=energy_cols)
    energy.loc[len(energy.index)] = energy_row
    return energy
