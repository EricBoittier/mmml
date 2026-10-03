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

"""Functions to set up and configure harmonic restraints.

.. deprecated::
    This module is deprecated. Use :mod:`pycharmm.restraints` instead:

    - ``cons_harm.setup_absolute()`` → ``restraints.atoms.harmonic_absolute()``
    - ``cons_harm.setup_best_fit()`` → ``restraints.atoms.harmonic_best_fit()``
    - ``cons_harm.setup_relative()`` → ``restraints.atoms.harmonic_relative()``
    - ``cons_harm.setup_pca()`` → ``restraints.positions.harmonic_pca()``
    - ``cons_harm.turn_off()`` → ``restraints.atoms.harmonic_turn_off()``

Corresponds to CHARMM command `CONS HARMonic`
See [CONS HARMonic documentation](https://academiccharmm.org/documentation/version/c47b1/cons#HarmonicAtom)

Functions
=========
- `setup_absolute` -- restrain atoms to fixed reference positions
- `setup_best_fit` -- restrain atoms with best-fit superposition
- `setup_relative` -- restrain atoms relative to another selection
- `setup_pca` -- restrain atoms for PCA analysis
- `turn_off` -- turn off all harmonic constraints

Keyword Arguments
=================
Common keyword arguments for restraint functions:

- `force_const` : float - restraint force constant k (default: 0.0)
- `expo` : int - exponent on distance (default: 2 for harmonic)
- `x_scale` : float - scale factor for x component (default: 1.0)
- `y_scale` : float - scale factor for y component (default: 1.0)
- `z_scale` : float - scale factor for z component (default: 1.0)
- `q_mass` : int - if 1, multiply k by atom mass (default: 0)
- `q_weight` : int - if 1, use weight array for k(i) (default: 0)
- `q_no_rot` : int - if 1, disable rotational restraint (default: 0)
- `q_no_trans` : int - if 1, disable translational restraint (default: 0)

Examples
========
>>> import pycharmm
>>> import pycharmm.cons_harm as cons_harm

Setup absolute positional restraints on all atoms
>>> cons_harm.setup_absolute(force_const=10.0)

Setup best-fit restraints on backbone atoms
>>> bb = pycharmm.SelectAtoms(atom_type='CA')
>>> cons_harm.setup_best_fit(selection=bb, force_const=5.0)

Setup relative restraints between two groups
>>> sel1 = pycharmm.SelectAtoms(seg_id='PROA')
>>> sel2 = pycharmm.SelectAtoms(seg_id='PROB')
>>> cons_harm.setup_relative(sel1, sel2, force_const=10.0)

Turn off all harmonic restraints
>>> cons_harm.turn_off()
"""

import warnings

from pycharmm import restraints


def turn_off():
    """
    Turn off and clear settings for harmonic constraints

    .. deprecated::
        Use ``restraints.atoms.harmonic_turn_off()`` instead.

    Returns
    -------
    bool
        True if successful
    """
    warnings.warn(
        "cons_harm.turn_off() is deprecated, use restraints.atoms.harmonic_turn_off()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.harmonic_turn_off()


def setup_pca(selection=None, comparison=False, **kwargs):
    """Configure and turn on absolute harmonic constraints for the selected atoms

    .. deprecated::
        Use ``restraints.positions.harmonic_pca()`` instead.

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
    warnings.warn(
        "cons_harm.setup_pca() is deprecated, use restraints.positions.harmonic_pca()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.positions.harmonic_pca(
        selection=selection, comparison=comparison, **kwargs
    )


def setup_absolute(selection=None, comparison=False, **kwargs):
    """
    Configure and turn on absolute harmonic restraints for the selected atoms

    .. deprecated::
        Use ``restraints.atoms.harmonic_absolute()`` instead.

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
    warnings.warn(
        "cons_harm.setup_absolute() is deprecated, use restraints.atoms.harmonic_absolute()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.harmonic_absolute(
        selection=selection, comparison=comparison, **kwargs
    )


def setup_best_fit(selection=None, comparison=False, **kwargs):
    """Configure and turn on best fit harmonic restraints for the selected atoms

    .. deprecated::
        Use ``restraints.atoms.harmonic_best_fit()`` instead.

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
    warnings.warn(
        "cons_harm.setup_best_fit() is deprecated, use restraints.atoms.harmonic_best_fit()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.harmonic_best_fit(
        selection=selection, comparison=comparison, **kwargs
    )


def setup_relative(iselection, jselection, comparison=False, **kwargs):
    """Configure and turn on relative harmonic restraints for the selected atoms

    .. deprecated::
        Use ``restraints.atoms.harmonic_relative()`` instead.

    The two selections must have the same number of selected atoms, as they
    are paired element-wise for the relative restraint calculation.

    *Valid* key word arguments for settings include
    `q_no_rot`, `q_no_trans`, `q_mass`, `q_weight`, `force_const`

    Parameters
    ----------
    iselection : pycharmm.SelectAtoms
        first selection of atoms for restraints
    jselection : pycharmm.SelectAtoms
        second selection of atoms for restraints (must match iselection count)
    comparison : bool
        if true, apply restraints on comparison set instead of main set
    **kwargs : optional
        key word arguments for relative harmonic constraints

    Returns
    -------
    bool
        True if successful

    Raises
    ------
    ValueError
        if the number of selected atoms in iselection and jselection differ
    """
    warnings.warn(
        "cons_harm.setup_relative() is deprecated, use restraints.atoms.harmonic_relative()",
        DeprecationWarning,
        stacklevel=2
    )
    return restraints.atoms.harmonic_relative(
        selection1=iselection, selection2=jselection, comparison=comparison, **kwargs
    )
