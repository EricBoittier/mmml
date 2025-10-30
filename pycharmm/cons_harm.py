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
Functions to set up and configure harmonic restraints.

Corresponds to CHARMM command `CONS HARMonic`  
See [CONS HARMonic documentation](https://academiccharmm.org/documentation/version/c47b1/cons#HarmonicAtom)

Examples
========
>>> import pycharmm.cons_harm as cons_harm
>>> import pycharmm.selection as selection
>>> import pycharmm.psf as psf
>>> cons_harm.setup_best_fit(force_const=10.0)
>>> natom = psf.get_natom()
>>> isel = select.none_selection(natom)
>>> isel = select.by_elt(range(0, 5), isel)
>>> isel = pycharmm.SelectAtoms().set_selection(isel)
>>> jsel = select.none_selection(natom)
>>> jsel = select.by_elt(range(5, 10), jsel)
>>> jsel = pycharmm.SelectAtoms().set_selection(jsel)
>>> cons_harm.setup_relative(isel, jsel, force_const=10.0)
"""

import ctypes

import pycharmm
import pycharmm.lib as lib


_OPTIONS_fields = [('expo', ctypes.c_int),
                   ('x_scale', ctypes.c_double),
                   ('y_scale', ctypes.c_double),
                   ('z_scale', ctypes.c_double),
                   ('q_no_rot', ctypes.c_int),
                   ('q_no_trans', ctypes.c_int),
                   ('q_mass', ctypes.c_int),
                   ('q_weight', ctypes.c_int),
                   ('force_const', ctypes.c_double)]


class _OPTIONS(ctypes.Structure):
    """A ctypes struct to hold harmonic constraint settings

    Attributes
    ----------
    expo : int 
        exponent on diff between atom and ref atom
    x_scale : float
        global scale factor for the x component
    y_scale : float
        global scale factor for the y component
    z_scale : float
        global scale factor for the z component
    q_no_rot : int
        do not do rotational restraint
    q_no_trans : int
        do not do translational restraint
    q_mass : int
        multiply k by atom mass (natural freq of oscillation of sqrt(k))
    q_weight : int
        use weight array for k(i) and not force argument above
    force_const : float, default = 0.0
        restraint force constant k
    """
    _fields_ = _OPTIONS_fields


_OPTIONS_defaults = (2, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0)

# TODO:
# 1. separate fill structure routine in util.py
# 2. add more error checking (eg relative:
#    check that sum(iselect) == sum(jselect))


def turn_off():
    """
    Turn off and clear settings for harmonic constraints

    Returns
    -------
    bool
        True if successful
    """
    status = lib.charmm.cons_harm_turn_off()
    bool(status)
    return status


def _make_opts(fields, settings):
    """Make a new instance of _OPTIONS

    Parameters
    ----------
    fields : list[string]
        list of valid field names used to filter settings
    settings : python dictionary
        name and value for harmonic restraints settings

    Returns
    -------
    _OPTIONS 
        an instance filled with values from settings
    """
    new_opts = _OPTIONS(*_OPTIONS_defaults)
    fields_types = dict(_OPTIONS_fields)
    valid_settings = [(k, v) for k, v in settings.items() if k in fields]
    for k, v in valid_settings:
        setattr(new_opts, k, fields_types[k](v))  # TODO: need a try/catch here

    return new_opts


def setup_pca(selection=None, comparison=False, **kwargs):
    """Configure and turn on absolute harmonic constraints for the selected atoms

    *Valid* key word arguments for settings include
    `expo`, `x_scale`, `y_scale`, `z_scale`, `q_mass`, `q_weight`, and `force_const`

    Parameters
    ----------
    selection : pycharmm.SelectAtoms, default = None
        apply restraints to selected atoms; None -> all atoms
    comparison : bool, default = False
        if true, apply restraints on comparison set instead of main set
    **kwargs : optional
        key word arguments for absolute harmonic constraints
    
    Returns
    -------
    bool
        True if successful
    """
    if not selection:
        selection = pycharmm.SelectAtoms().all_atoms()

    abs_fields = ['expo', 'x_scale', 'y_scale', 'z_scale',
                  'q_mass', 'q_weight', 'force_const']
    opts = _make_opts(abs_fields, kwargs)

    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)
    status = lib.charmm.cons_harm_setup_pca(c_sel,
                                            ctypes.byref(c_comp),
                                            ctypes.byref(opts))
    status = bool(status)
    return status


def setup_absolute(selection=None, comparison=False, **kwargs):
    """
    Configure and turn on absolute harmonic restraints for the selected atoms

    *Valid* key word arguments for settings include
    `expo`, `x_scale`, `y_scale`, `z_scale`, `q_mass`, `q_weight`, and `force_const`

    Parameters
    ----------
    selection : pycharmm.SelectAtoms 
        apply restraints to selected atoms; None -> all atoms
    comparison : bool, default = False
        if true, do restraints on comparison set instead of main set
    **kwargs : optional
        key word arguments for absolute harmonic constraints

    Returns 
    -------
    bool
        True if successful
    """
    if not selection:
        selection = pycharmm.SelectAtoms().all_atoms()

    abs_fields = ['expo', 'x_scale', 'y_scale', 'z_scale',
                  'q_mass', 'q_weight', 'force_const']
    opts = _make_opts(abs_fields, kwargs)

    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)
    status = lib.charmm.cons_harm_setup_absolute(c_sel,
                                                 ctypes.byref(c_comp),
                                                 ctypes.byref(opts))
    status = bool(status)
    return status


def setup_best_fit(selection=None, comparison=False, **kwargs):
    """Configure and turn on best fit harmonic restraints for the selected atoms

    *Valid* key word arguments for settings include
    `q_no_rot`, `q_no_trans`, `q_mass`, `q_weight`, `force_const`

    Parameters
    ----------
    selection : pycharmm.SelectAtoms 
        apply restraints to selected atoms
    comparison : bool
        if true, do restraints on comparison set instead of main set
    **kwargs : optional
        key word arguments for best fit harmonic constraints

    Returns
    -------
    bool
        True if successful
    """
    if not selection:
        selection = pycharmm.SelectAtoms().all_atoms()

    best_fit_fields = ['q_no_rot', 'q_no_trans',
                       'q_mass', 'q_weight', 'force_const']
    opts = _make_opts(best_fit_fields, kwargs)

    c_sel = selection.as_ctypes()
    c_comp = ctypes.c_int(comparison)
    status = lib.charmm.cons_harm_setup_best_fit(c_sel,
                                                 ctypes.byref(c_comp),
                                                 ctypes.byref(opts))
    status = bool(status)
    return status


def setup_relative(iselection, jselection, comparison=False, **kwargs):
    """Configure and turn on relative harmonic restraintfor the selected atoms

    *Valid* key word arguments for settings include
    `q_no_rot`, `q_no_trans`, `q_mass`, `q_weight`, `force_const`

    Parameters
    ----------
    iselection : pycharmm.SelectAtoms
        apply restraints to selected atoms
    jselection : pycharmm.SelectAtoms  
        apply restraints to selected atoms
    comparison : bool
        if true, apply restraints on comparison set instead of main set
    **kwargs : optional 
        key word arguments for relative harmonic constraints
    Returns
    -------
    bool
        True if successful
    """
    relative_fields = ['q_no_rot', 'q_no_trans',
                       'q_mass', 'q_weight', 'force_const']
    opts = _make_opts(relative_fields, kwargs)

    c_isel = iselection.as_ctypes()
    c_jsel = jselection.as_ctypes()
    c_comp = ctypes.c_int(comparison)
    status = lib.charmm.cons_harm_setup_relative(c_isel, c_jsel,
                                                 ctypes.byref(c_comp),
                                                 ctypes.byref(opts))
    status = bool(status)
    return status
