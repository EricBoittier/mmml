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

"""Functions to manipulate scalar atom properties

Corresponds to CHARMM command `SCALar`

See CHARMM documentation [scalar](<https://academiccharmm.org/documentation/version/c47b1/scalar>)
for more information

Functions
=========
- `get_fbetas` -- get friction coefficients for the atoms
- `set_fbetas` -- set friction coefficients for the selected atoms
- `get_charges` -- get charges for the atoms
- `set_charges` -- set charges for the selected atoms
- `get_masses` -- get masses for the atoms
- `set_masses` -- set masses for the selected atoms
- `get_econt` -- get the energy partition array
- `set_econt` -- for selected atoms, update the energy partition array
- `get_epcont` -- get the free energy difference atom partition
- `set_epcont` -- for selected atoms, update the free energy difference atom partition
- `get_constraints` -- get the harmonic restraint constants
- `set_constraints` -- for selected atoms, update the harmonic restraint constants
- `get_move` -- get the flags indicating which atoms move
- `set_move` -- for selected atoms, update the flags indicating which atoms move
- `get_ignore` -- get the ASP flags indicating which atoms are ignored
- `set_ignore` -- for selected atoms, update the ASP flags indicating which atoms are ignored
- `get_aspv` -- get atomic solvation parameter (ASP) value
- `set_aspv` -- for selected atoms, update atomic solvation parameter (ASP) value
- `get_vdw_surf` -- get vdw radius for ASP solvation energy, includes probe radius
- `set_vdw_surf` -- for select atoms, update vdw radius for ASP solvation ener, includes probe radius
- `get_rscale` -- get radius scale factor for nonbonded (vdw)
- `set_rscale` -- for select atoms, update radius scale factor for nonbonded (vdw)
- `get_wcad` -- get Weeks, Chandler, Anderson decomp of Lennard-Jones Potential
- `set_wcad` -- for select atoms, update Weeks, Chandler, Anderson LJ Potential decomp
- `get_alpha` -- get atom polarizability
- `get_effect` -- get effective number of electrons
- `get_radius` -- get van der Waals radii
- `get_fqprin` -- get FQ Slater orbital principal quantum number
- `get_fqzeta` -- get FQ Slater orbital exponent
- `get_fqchi` -- get FQ electronegativity parameter
- `get_fqmass` -- get FQ charge mass
- `get_fqjz` -- get FQ self-interaction
- `get_fqcforce` -- get FQ charge force
- `set_fqcforce` -- for select atoms, update FQ charge force
- `get_fqold` -- get FQ charges from last timestep
- `set_fqold` -- for select atoms, update FQ charges from last timestep
- `get_varc` -- get variable cutoffs of LJ interaction depending on atom types
- `set_varc` -- for select atoms, set variable cutoffs of LJ interaction
- `get_sgwt` -- get self-guiding weights for SGLD simulation
- `set_sgwt` -- for select atoms, set self-guiding weights for SGLD simulation
- `get_sggamma` -- get apparent friction constants for SGMD/SGLD simulation
- `set_sggamma` -- for select atoms, set apparent friction constants for SGMD/SGLD
- `get` -- get scalar atom properties

Examples
========
Set fbeta of all atoms to 1.0 
>>> import pycharmm.psf as psf
>>> import pycharmm.scalar as scalar
>>> n = psf.get_natom()
>>> scalar.set_fbetas([1.0] * n)

"""

import ctypes

import pycharmm
import pycharmm.lib as lib
import pycharmm.psf as psf


def get_charges():
    """Get charges for the atoms

    Returns
    -------
    charges : list[float]
        charges for the atoms
    """
    n = psf.get_natom()
    charges = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_charges(charges)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting charges.')

    return list(charges)


def set_charges(charges, selection=None):
    """Set charges for the selected atoms

    Parameters
    ----------
    charges : list[float]
        charges for each atom
    selection: pycharmm.SelectAtoms
        set charges for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_charges routine
    """
    n = psf.get_natom()
    charges = (ctypes.c_double * n)(*charges)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_charges(charges, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting charges.')

    return int(status)


def get_masses():
    """Get masses for the atoms

    Returns
    -------
    masses : list[float]
        masses for the atoms
    """
    n = psf.get_natom()
    masses = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_masses(masses)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting masses.')

    return list(masses)


def set_masses(masses, selection=None):
    """Set masses for the selected atoms

    Parameters
    ----------
    masses : list[float]
        masses for each atom
    selection: pycharmm.SelectAtoms
        set masses for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_masses routine
    """
    n = psf.get_natom()
    masses = (ctypes.c_double * n)(*masses)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_masses(masses, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting masses.')

    return int(status)


def get_fbetas():
    """Get friction coefficients for the atoms

    Returns
    -------
    fbetas : list[float]
        friction coefficients for the atoms
    """
    n = psf.get_natom()
    fbetas = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fbetas(fbetas)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fbetas.')

    return list(fbetas)


def set_fbetas(fbetas, selection=None):
    """Set friction coefficients for the selected atoms

    Parameters
    ----------
    fbetas : list[float]
         friction coefficients for each atom
    selection: pycharmm.SelectAtoms
         set friction coefficients for only these selected atoms

         default None results in all atoms selected

    Returns
    -------
    int
           status code returned by lib.charmm.scalar_set_fbetas routine
    """
    n = psf.get_natom()
    fbetas = (ctypes.c_double * n)(*fbetas)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_fbetas(fbetas, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting fbetas.')

    return int(status)


def get_econt():
    """Get the energy partition array

    Returns
    -------
    list[float]
        the energy partition array
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_econt(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting econt.')

    return list(old_values)


def set_econt(new_vals, selection=None):
    """For selected atoms, update the energy partition array

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection : pycharmm.SelectAtoms 
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int 
        status code returned by lib.charmm.scalar_set_econt routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_econt(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting econt.')

    return int(status)


def get_epcont():
    """Get the free energy difference atom partition

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_epcont(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting epcont.')

    return list(old_values)


def set_epcont(new_vals, selection=None):
    """For selected atoms, update the free energy difference atom partition

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int 
        status code returned by lib.charmm.scalar_set_epcont API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_epcont(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting epcont.')

    return int(status)


def get_constraints():
    """Get the harmonic restraint constants

    Returns
    -------
    list[float] 
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_constraints(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting constraints.')

    return list(old_values)


def set_constraints(new_vals, selection=None):
    """For selected atoms, update the harmonic restraint constants

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int 
        status code returned by lib.charmm.scalar_set_constraints API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_constraints(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting constraints.')

    return int(status)


def get_move():
    """Get the flags indicating which atoms move

    Returns
    -------
    list[int]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_int * n)(0)
    status = lib.charmm.scalar_get_move(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting move.')

    return list(old_values)


def set_move(new_vals, selection=None):
    """For selected atoms, update the flags indicating which atoms move

    Parameters
    ----------
    new_vals: list[int]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    status : list[int]
        status code returned by lib.charmm.scalar_set_move API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_int * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_move(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting move.')

    return int(status)


def get_ignore():
    """Get the ASP flags indicating which atoms are ignored

    Returns
    -------
    list[int]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_int * n)(0)
    status = lib.charmm.scalar_get_ignore(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting ignore.')

    return list(old_values)


def set_ignore(new_vals, selection=None):
    """For selected atoms, update the ASP flags indicating which atoms are ignored

    Parameters
    ----------
    new_vals: list[int]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_ignore API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_int * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_ignore(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting ignore.')

    return int(status)


def get_aspv():
    """Get ASP parameter value

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_aspv(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting aspv.')

    return list(old_values)


def set_aspv(new_vals, selection=None):
    """For selected atoms, update ASP parameter values 

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_aspv API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_aspv(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting aspv.')

    return int(status)


def get_vdw_surf():
    """Get vdw radius for ASP solvation energy, includes probe radius

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_vdw_surf(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting vdw_surf.')

    return list(old_values)


def set_vdw_surf(new_vals, selection=None):
    """For select atoms, update vdw radius for ASP solvation energy, includes probe radius

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_vdw_surf routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_vdw_surf(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting vdw_surf.')

    return int(status)


def get_rscale():
    """Get radius scale factor for nonbonded (vdw)

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_rscale(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting rscale.')

    return list(old_values)


def set_rscale(new_vals, selection=None):
    """For select atoms, update radius scale factor for nonbonded (vdw)

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms
        
        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by lib.charmm.scalar_set_rscale API routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_rscale(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting rscale.')

    return int(status)


def get_wcad():
    """Get Weeks, Chandler, Anderson decomp of Lennard-Jones Potential

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_wcad(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting wcad.')

    return list(old_values)


def set_wcad(new_vals, selection=None):
    """For select atoms, update Weeks, Chandler, Anderson LJ Potential decomp

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int 
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_wcad(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting wcad.')

    return int(status)


def get_alpha():
    """Get atom polarizability

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_alpha(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting alpha.')

    return list(old_values)


def get_effect():
    """Get effective number of electrons

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_effect(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting effect.')

    return list(old_values)


def get_radius():
    """Get van der Waals radii

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_radius(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting radius.')

    return list(old_values)


def get_fqprin():
    """Get FQ Slater orbital principal quantum number

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqprin(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqprin.')

    return list(old_values)


def get_fqzeta():
    """Get FQ Slater orbital exponent

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqzeta(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqzeta.')

    return list(old_values)


def get_fqchi():
    """Get FQ electronegativity parameter

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqchi(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqchi.')

    return list(old_values)


def get_fqmass():
    """Get FQ charge mass

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqmass(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqmass.')

    return list(old_values)


def get_fqjz():
    """Get FQ self-interaction

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqjz(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqjz.')

    return list(old_values)


def get_fqcforce():
    """Get FQ charge force

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqcforce(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqcforce.')

    return list(old_values)


def set_fqcforce(new_vals, selection=None):
    """For select atoms, update FQ charge force

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_fqcforce(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting fqcforce.')

    return int(status)


def get_fqold():
    """Get FQ charges from last timestep

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_fqold(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting fqold.')

    return list(old_values)


def set_fqold(new_vals, selection=None):
    """For select atoms, update FQ charges from last timestep

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected
     
    Returns
    -------
    int
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_fqold(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting fqold.')

    return int(status)


def get_varc():
    """Get variable cutoffs of LJ interaction depending on atom types

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_varc(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting varc.')

    return list(old_values)


def set_varc(new_vals, selection=None):
    """For select atoms, set variable cutoffs of LJ interaction

    depends on atom type

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_varc(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting varc.')

    return int(status)


def get_sgwt():
    """Get self-guiding weights for SGLD simulation

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_sgwt(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting sgwt.')

    return list(old_values)


def set_sgwt(new_vals, selection=None):
    """For select atoms, set self-guiding weights for SGLD simulation

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_sgwt(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting sgwt.')

    return int(status)


def get_sggamma():
    """Get apparent friction constants for SGMD/SGLD simulation

    Returns
    -------
    list[float]
        array of scalar values
    """
    n = psf.get_natom()
    old_values = (ctypes.c_double * n)(0)
    status = lib.charmm.scalar_get_sggamma(old_values)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem getting sggamma.')

    return list(old_values)


def set_sggamma(new_vals, selection=None):
    """For select atoms, set apparent friction constants for SGMD/SGLD

    Parameters
    ----------
    new_vals: list[float]
        new values to set
    selection: pycharmm.SelectAtoms
        set new values for only these selected atoms

        default None results in all atoms selected

    Returns
    -------
    int
        status code returned by charmm's scalar set api routine
    """
    n = psf.get_natom()
    new_vals = (ctypes.c_double * n)(*new_vals)

    if selection is None:
        selection = pycharmm.SelectAtoms().all_atoms()

    c_sel = selection.as_ctypes()

    status = lib.charmm.scalar_set_sggamma(new_vals, c_sel)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting sggamma.')

    return int(status)


def get(fbeta=False, econt=False, epcont=False, constraints=False,
        move=False, ignore=False,
        aspv=False, vdw_surf=False, rscale=False, wcad=False,
        alpha=False, effect=False, radius=False,
        fqprin=False, fqzeta=False, fqchi=False, fqmass=False,
        fqjz=False, fqcforce=False, fqold=False,
        varc=False,
        sgwt=False, sggamma=False):
    """Get scalar atom properties in a dictionary

    Parameters
    ----------
    fbeta:  bool
        include friction coeff for each atom?
    econt:  bool
        include energy partion array?
    epcont:  bool
        include energy difference atom partion array?
    constraints:  bool
        include harmonic contraint constants?
    move: bool
        include flags indicating which atoms move?
    ignore:  bool
        include flags indicating which atoms are ignored?
    aspv:  bool
        include atomic solvation parameters?
    vdw_surf:  bool
        include vdw radius for solvent energy?
    rscale:  bool
        include radius scale factor for nonbonded (vdw)?
    wcad:  bool
        include Weeks, Chandler, Anderson LJ Potential decomp?
    alpha: bool
        include atom polarizability?
    effect:  bool
        include effective number of electrons?
    radius: bool
        include van der Waals radii?
    fqprin: bool
        include FQ Slater orbital principal quantum number?
    fqzeta: bool
        include FQ Slater orbital exponent?
    fqchi: bool
        include FQ electronegativity parameter?
    fqmass: bool
        include FQ charge mass?
    fqjz: bool
        include FQ self-interaction?
    fqcforce: bool
        include FQ charge force?
    fqold:  bool
        include FQ charges from last timestep?
    varc: bool
        include variable cutoffs of LJ interaction?
    sgwt: bool
        include self-guiding weights for SGLD simulation?
    sggamma: bool
        include apparent friction constants for SGMD/SGLD?

    Returns
    -------
    dict
        a dictionary whose keys are True keyword arguments, 
        and values are list of corresponding scalar values
    """
    n = psf.get_natom()
    scalars = dict()

    if fbeta:
        vals = get_fbetas()
        scalars['fbeta'] = vals[:n]

    if econt:
        vals = get_econt()
        scalars['econt'] = vals[:n]

    if epcont:
        vals = get_epcont()
        scalars['epcont'] = vals[:n]

    if constraints:
        vals = get_constraints()
        scalars['constraints'] = vals[:n]

    if move:
        vals = get_move()
        scalars['move'] = vals[:n]

    if ignore:
        vals = get_ignore()
        scalars['ignore'] = vals[:n]

    if aspv:
        vals = get_aspv()
        scalars['aspv'] = vals[:n]

    if vdw_surf:
        vals = get_vdw_surf()
        scalars['vdw_surf'] = vals[:n]

    if rscale:
        vals = get_rscale()
        scalars['rscale'] = vals[:n]

    if wcad:
        vals = get_wcad()
        scalars['wcad'] = vals[:n]

    if alpha:
        vals = get_alpha()
        scalars['alpha'] = vals[:n]

    if effect:
        vals = get_effect()
        scalars['effect'] = vals[:n]

    if radius:
        vals = get_radius()
        scalars['radius'] = vals[:n]

    if fqprin:
        vals = get_fqprin()
        scalars['fqprin'] = vals[:n]

    if fqzeta:
        vals = get_fqzeta()
        scalars['fqzeta'] = vals[:n]

    if fqchi:
        vals = get_fqchi()
        scalars['fqchi'] = vals[:n]

    if fqmass:
        vals = get_fqmass()
        scalars['fqmass'] = vals[:n]

    if fqjz:
        vals = get_fqjz()
        scalars['fqjz'] = vals[:n]

    if fqcforce:
        vals = get_fqcforce()
        scalars['fqcforce'] = vals[:n]

    if fqold:
        vals = get_fqold()
        scalars['fqold'] = vals[:n]

    if varc:
        vals = get_varc()
        scalars['varc'] = vals[:n]

    if sgwt:
        vals = get_sgwt()
        scalars['sgwt'] = vals[:n]

    if sggamma:
        vals = get_sggamma()
        scalars['sggamma'] = vals[:n]

    return scalars
