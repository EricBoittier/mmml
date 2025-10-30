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

"""Functions to configure and run molecular dynamics

Corresponds to CHARMM command `DYNAmics`
  
See CHARMM documentation [dynamc](<https://academiccharmm.org/documentation/version/c47b1/dynamc>)
for more information


Functions
=========
- `use_lang` -- use Langevin dynamics
- `use_start` -- use starting velocities described by iasvel
- `set_nprint` -- change step freq to print and store energy data during runs
- `get_nprint` -- return step freq to print and store energy data during runs
- `set_nstep` -- change the number of steps to be taken in each dynamics run
- `get_nstep` -- return the number of steps to be taken in each dynamics run
- `set_inbfrq` -- change the freq of regenerating nonbonded list during runs
- `set_ihbfrq` -- change the freq of regenerating the hydrogen bond list
- `set_timest` -- set the time step for dynamics
- `set_akmast` -- set the time step for dynamics in AKMA units
- `set_firstt` -- set the initial temperature for dynamics runs
- `run` -- run the dynamics

Examples
========
A simple NVT simulation using Langevin dynamics at 298.15 K
>>> import pycharmm
>>> import pycharmm.psf as psf
>>> import pycharmm.scalar as scalar
>>> n = psf.get_natom()
>>> scalar.set_fbetas([1.0] * n)
>>> prod = pycharmm.DynamicsScript(start = True, leap = True, verlet = False, cpt = False, new = False, langevin = True, omm = False, timestep = 0.002, nstep = 50000, nsavc = 5000, nsavv = 0, nsavl = 0,  nprint = 1000, iprfrq = 1000, isvfrq = 1000, ntrfrq = 5000, inbfrq = -1, ihbfrq = 0, ilbfrq = 0, imgfrq = -1, iunrea = -1, iunwri = 70, iuncrd = -1, iunldm = -1, firstt = 298.15, finalt = 298.15, tstruct = 298.15, tbath = 298.15, iasors = 1, iasvel = 1, iscale = 0, scale = 1, ichecw = 0, echeck = -1)
>>> prod.run()

"""

import ctypes

import pandas

import pycharmm.lib as lib
import pycharmm.coor as coor
import pycharmm.psf as psf
import pycharmm.script as script


# TODO:
# 1. change all functions to keyword style signatures
# 2. take care of comment formatting
# 3. take care of any pycharm errors


class OPTIONS(ctypes.Structure):
    """A ctypes struct to hold runtime dynamics settings

    Attributes
    ----------
    ieqfrq : int
        The step frequency for assigning or scaling velocities to
        FINALT temperature during the equilibration stage of the
        dynamics run.
    ntrfrq : int
        The step frequency for stopping the rotation and translation
        of the molecule during dynamics. This operation is done
        automatically after any heating.
    ichecw : int
        The option for checking to see if the average temperature
        of the system lies within the allotted temperature window
        (between FINALT+TWINDH and FINALT+TWINDL) every
        IEQFRQ steps.

        .eq. 0 - do not check,
                 i.e., assign or scale velocities.

        .ne. 0 - check window,
                 i.e., assign or scale velocities only if average
                 temperature lies outside the window.

    """
    _fields_ = [('ieqfrq', ctypes.c_int),
                ('ntrfrq', ctypes.c_int),
                ('ichecw', ctypes.c_int),
                ('tbath', ctypes.c_double),
                ('iasors', ctypes.c_int),
                ('iasvel', ctypes.c_int),
                ('iscale', ctypes.c_int),
                ('iscvel', ctypes.c_int),
                ('isvfrq', ctypes.c_int),
                ('iprfrq', ctypes.c_int),
                ('ihtfrq', ctypes.c_int)]


def use_lang():
    """Use Langevin dynamics

    Returns
    -------
    int
        True if Langevin dynamics was already selected
    """
    old_lang = lib.charmm.dynamics_use_lang()
    old_lang = bool(old_lang)
    return old_lang


def use_start():
    """Use starting velocities described by iasvel

    Returns
    -------
    bool
        True if start was already selected
    """
    old_start = lib.charmm.dynamics_use_start()
    old_start = bool(old_start)
    return old_start


def use_restart():
    """Dynamics is restarted by reading restart file from iunrea

    Returns
    -------
    bool
        True if start was already selected
    """
    old_restart = lib.charmm.dynamics_use_restart()
    old_restart = bool(old_restart)
    return old_restart


def set_nprint(new_nprint):
    """Change step freq for printing and storing energy data for dynamics runs

    Parameters
    ----------
    new_nprint: int
        the new step frequency desired

    Returns
    -------
    int
        old step freq
    """
    new_nprint = ctypes.c_int(new_nprint)
    old_nprint = lib.charmm.dynamics_set_nprint(ctypes.byref(new_nprint))
    return old_nprint


def get_nprint():
    """Return step freq for printing and storing energy data for dynamics runs

    Returns
    -------
    int
        the current step frequency
    """
    nprint = lib.charmm.dynamics_get_nprint()
    return int(nprint)


def set_nstep(new_nstep):
    """Change the number of steps to be taken in each dynamics run

    changes the number of dynamics steps which is equal to
    the number of energy evaluations

    Parameters
    ----------
    new_nstep: int
        the new number of dynamics steps desired

    Returns
    -------
    int
        the previous setting for the number of dynamics steps
    """
    new_nstep = ctypes.c_int(new_nstep)
    old_nstep = lib.charmm.dynamics_set_nstep(ctypes.byref(new_nstep))
    return old_nstep


def get_nstep():
    """Return the number of steps to be taken in each dynamics run

    returns the number of dynamics steps which is equal to
    the number of energy evaluations

    Returns
    -------
    int
        the current number of steps to be taken
    """
    nstep = lib.charmm.dynamics_get_nstep()
    return int(nstep)


def set_inbfrq(new_inbfrq):
    """Change the freq of regenerating the nonbonded list for dynamics runs

    The list is regenerated if the current step number
    modulo INBFRQ is zero and if INBFRQ is non-zero.

    Specifying zero prevents the non-bonded list from being
    regenerated at all.

    INBFRQ = -1 --> all lists are updated when necessary
    (heuristic test).

    Parameters
    ----------
    new_inbfrq: int
        the new freq for nonbonded list regeneration

    Returns
    -------
    int
        the old inbfrq
    """
    new_inbfrq = ctypes.c_int(new_inbfrq)
    old_inbfrq = lib.charmm.dynamics_set_inbfrq(ctypes.byref(new_inbfrq))
    return old_inbfrq


def set_ihbfrq(new_ihbfrq):
    """Change the freq of regenerating the hydrogen bond list

    analogous to set_inbfrq

    Parameters
    ----------
    new_ihbfrq: int
        the new freq for hydrogen bond list regeneration

    Returns
    -------
    int
        the old inbfrq
    """
    new_ihbfrq = ctypes.c_int(new_ihbfrq)
    old_ihbfrq = lib.charmm.dynamics_set_ihbfrq(ctypes.byref(new_ihbfrq))
    return old_ihbfrq


def set_ilbfrq(new_ilbfrq):
    """Change the freq of checking whether an atom is in the Langevin region

    Langevin region defined by RBUF

    Parameters
    ----------
    new_ilbfrq: int
        the new freq
    Returns
    -------
    int
        the old freq
    """
    new_ilbfrq = ctypes.c_int(new_ilbfrq)
    old_ilbfrq = lib.charmm.dynamics_set_ilbfrq(ctypes.byref(new_ilbfrq))
    return old_ilbfrq


def set_finalt(new_finalt):
    """Set the final equilibrium temperature

    important for all stages except initiation

    the default is 298.0 Kelvin

    Parameters
    ----------
    new_finalt : float
        new final temperature in Kelvin

    Returns
    -------
    float 
        old final temperature in Kelvin
    """
    new_finalt = ctypes.c_double(new_finalt)
    charmm_set_finalt = lib.charmm.dynamics_set_finalt
    charmm_set_finalt.restype = ctypes.c_double
    old_finalt = charmm_set_finalt(ctypes.byref(new_finalt))
    return old_finalt


def set_teminc(new_teminc):
    """Set the temperature increment to be given to the system every IHTFRQ steps

    important for the heating stage

    the default is 5.0 Kelvin

    Parameters
    ----------
    new_teminc: float
         the new temperature increment

    Returns
    -------
    float
         the old temperature increment
    """
    new_teminc = ctypes.c_double(new_teminc)
    charmm_set_teminc = lib.charmm.dynamics_set_teminc
    charmm_set_teminc.restype = ctypes.c_double
    old_teminc = charmm_set_teminc(ctypes.byref(new_teminc))
    return old_teminc


def set_tstruc(new_tstruc):
    """Set the temperature at which the starting structure has been equilibrated

    used to assign velocities so that equal
    partition of energy will yield the correct equilibrated
    temperature

    -999.0 is a default which causes the
    program to assign velocities at T = 1.25 * FIRSTT

    Parameters
    ----------
    new_tstruc : float
        the new temperature

    Returns
    -------
    float : 
        the old temperature
    """
    new_tstruc = ctypes.c_double(new_tstruc)
    charmm_set_tstruc = lib.charmm.dynamics_set_tstruc
    charmm_set_tstruc.restype = ctypes.c_double
    old_tstruc = charmm_set_tstruc(ctypes.byref(new_tstruc))
    return old_tstruc


def get_nrand():
    """Return the number of integers required to seed the random number generator

    the random number generator plays a role in assigning velocities

    Returns
    -------
    int
         the current number of seed integers the random number generator needs
    """
    nrand = lib.charmm.dynamics_get_nrand()
    return int(nrand)


def set_rngseeds(new_seeds):
    """Set seed for the random number generator

    The seed for the random number generator used for
    assigning velocities. If not specified a value based on
    the system clock is used; this is the recommended mode, since
    it makes each run unique.

    One integer, or as many as required by the random number
    generator, may be specified. See CHARMM documentation
    [random](<https://academiccharmm.org/documentation/version/c47b1/random>)

    Parameters
    ----------
    new_seeds: list[int]
        new seeds for the random number generator

    Returns
    -------
    bool
        success == True
    """
    nrand = get_nrand()
    typed_seeds = (ctypes.c_int * nrand)(*new_seeds)
    success = lib.charmm.dynamics_set_rngseeds(typed_seeds)
    success = bool(success)
    return success


def set_iseed(new_seeds):
    """An alias for set_rngseeds"""
    return set_rngseeds(new_seeds)


def set_timest(new_timest):
    """Set the time step in picoseconds for dynamics

    the default is 0.001 picoseconds

    Parameters
    ----------
    new_timest: float
        the new time step in picoseconds

    Returns
    -------
    float
        old time step in picoseconds
    """
    new_timest = ctypes.c_double(new_timest)
    charmm_set_timest = lib.charmm.dynamics_set_timest
    charmm_set_timest.restype = ctypes.c_double
    old_timest = charmm_set_timest(ctypes.byref(new_timest))
    return old_timest


def set_akmast(new_akmast):
    """Set the time step for dynamics in AKMA units

    Parameters
    ----------
    new_akmast:  float
        the new time step in AKMA units

    Returns
    -------
    float
        old time step in AKMA units
    """
    new_akmast = ctypes.c_double(new_akmast)
    charmm_set_akmast = lib.charmm.dynamics_set_akmast
    charmm_set_akmast.restype = ctypes.c_double
    old_akmast = charmm_set_akmast(ctypes.byref(new_akmast))
    return old_akmast


def set_firstt(new_firstt):
    """Set the initial temperature for dynamics runs

    Set the initial temperature at which the velocities have to be
    assigned to begin the dynamics run. Important only
    for the initial stage of a dynamics run.

    Parameters
    ----------
    new_firstt:  float
        the initial temperature desired

    Returns
    -------
    float
        old initial temp
    """
    new_firstt = ctypes.c_double(new_firstt)
    charmm_set_firstt = lib.charmm.dynamics_set_firstt
    charmm_set_firstt.restype = ctypes.c_double
    old_firstt = charmm_set_firstt(ctypes.byref(new_firstt))
    return old_firstt


def set_twindh(new_twindh):
    """Set the high temperature tolerance for equilibration

    Parameters
    ----------
    new_twindh: float
        the new high temp tol for equilibration

    Returns
    -------
    float
        old high temp tol for equilibration
    """
    new_twindh = ctypes.c_double(new_twindh)
    charmm_set_twindh = lib.charmm.dynamics_set_twindh
    charmm_set_twindh.restype = ctypes.c_double
    old_twindh = charmm_set_twindh(ctypes.byref(new_twindh))
    return old_twindh


def set_twindl(new_twindl):
    """Set the low temperature tolerance for equilibration

    Parameters
    ----------
    new_twindl: float
        the new low temp tol for equilibration

    Returns
    -------
    float
        old low temp tol for equilibration
    """
    new_twindl = ctypes.c_double(new_twindl)
    charmm_set_twindl = lib.charmm.dynamics_set_twindl
    charmm_set_twindl.restype = ctypes.c_double
    old_twindl = charmm_set_twindl(ctypes.byref(new_twindl))
    return old_twindl


def set_echeck(new_echeck):
    """Set the total energy change tolerance for each step

    Parameters
    ----------
    new_echeck: float
        the new energy change tolerance

    Returns
    -------
    float
        old energy change tolerance
    """
    new_echeck = ctypes.c_double(new_echeck)
    charmm_set_echeck = lib.charmm.dynamics_set_echeck
    charmm_set_echeck.restype = ctypes.c_double
    old_echeck = charmm_set_echeck(ctypes.byref(new_echeck))
    return old_echeck


def set_nsavc(new_nsavc):
    """Set the freq for saving coords to file

    Parameters
    ----------
    new_nsavc: int
        the new frequency for saving coords to file

    Returns
    -------
    int
        the old frequency for saving coords to file
    """
    new_nsavc = ctypes.c_int(new_nsavc)
    old_nsavc = lib.charmm.dynamics_set_nsavc(ctypes.byref(new_nsavc))
    return old_nsavc


def set_nsavv(new_nsavv):
    """Set the frequence for saving velocities to file

    Parameters
    ----------
    new_nsavv: int
        the new frequency for seving velocities to file

    Returns
    -------
    int  
        the old frequency for seving velocities to file
    """
    new_nsavv = ctypes.c_int(new_nsavv)
    old_nsavv = lib.charmm.dynamics_set_nsavv(ctypes.byref(new_nsavv))
    return old_nsavv


def set_fbetas(fbetas):
    """Set friction coefficients for atoms for Langevin dynamics

    Parameters
    ----------
    fbetas: list[float]
        length natom, set atom i friction coefficient to fbetas[i]

    Returns
    -------
    list[float]
        old friction coefficients for the atoms
    """
    n = psf.get_natom()
    fbetas = (ctypes.c_double * n)(*fbetas)

    status = lib.charmm.dynamics_set_fbetas(fbetas)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting fbetas.')

    return list(fbetas)


def set_iunwri(filename):
    """Open a unit to use for writing the restart file

    Parameters
    ----------
    filename: string
        new file path to write

    Returns
    -------
    bool
        true if successful
    """
    fn = ctypes.c_char_p(filename.encode())
    len_fn = ctypes.c_int(len(filename))

    status = lib.charmm.dynamics_set_iunwri(fn, ctypes.byref(len_fn))

    status = bool(status)
    return status


def set_iuncrd(filename):
    """Open a unit to use for writing the coordinate file

    Parameters
    ----------
    filename: string
        new file path to write

    Returns
    -------
    bool
        true if successful
    """
    fn = ctypes.c_char_p(filename.encode())
    len_fn = ctypes.c_int(len(filename))

    status = lib.charmm.dynamics_set_iuncrd(fn, ctypes.byref(len_fn))

    status = bool(status)
    return status


def set_iunrea(filename):
    """Open a unit to read the restart file

    Parameters
    ----------
    filename:  string
        new file path to read

    Returns
    -------
    bool
        true if successful
    """
    fn = ctypes.c_char_p(filename.encode())
    len_fn = ctypes.c_int(len(filename))

    status = lib.charmm.dynamics_set_iunrea(fn, ctypes.byref(len_fn))

    status = bool(status)
    return status


def _configure(**kwargs):
    """Set dynamics parameters from a dictionary of names and values

    Parameters
    ----------
    **kwargs:
        names and values from OPTIONS

    Returns
    -------
    OPTIONS:
        a ctypes.Structure class for options that get set when
        dynamics runs, e.g. ieqfrq, ntrfrq and ichew
        this is to be passed into a call to the run function
    """
    valid_opts = dict(
        [('ieqfrq', 0),
         ('ntrfrq', 0),
         ('ichecw', 0),
         ('tbath', 298.0),
         ('iasors', 0),
         ('iasvel', 1),
         ('iscale', 0),
         ('iscvel', 0),
         ('isvfrq', 100),
         ('iprfrq', 100),
         ('ihtfrq', 0)])

    options = OPTIONS()
    for opt, default in valid_opts.items():
        setattr(options, opt, default)

    valid_toggles = dict(
        [('lang', use_lang),
         ('start', use_start),
         ('restart', use_restart)])

    valid_setters = dict([('nprint', set_nprint),
                          ('nstep',  set_nstep),
                          ('inbfrq', set_inbfrq),
                          ('ihbfrq', set_ihbfrq),
                          ('ilbfrq', set_ilbfrq),
                          ('finalt', set_finalt),
                          ('teminc', set_teminc),
                          ('tstruc', set_tstruc),
                          ('timest', set_timest),
                          ('akmast', set_akmast),
                          ('firstt', set_firstt),
                          ('twindh', set_twindh),
                          ('twindl', set_twindl),
                          ('echeck', set_echeck),
                          ('nsavc', set_nsavc),
                          ('nsavv', set_nsavv),
                          ('iseed', set_rngseeds),
                          ('fbeta', set_fbetas),
                          ('iunwri', set_iunwri),
                          ('iuncrd', set_iuncrd),
                          ('iunrea', set_iunrea)])

    for k, v in kwargs.items():
        if k in valid_opts:
            setattr(options, k, v)
        elif k in valid_toggles:
            toggle = valid_toggles[k]
            toggle()
        elif k in valid_setters:
            setter = valid_setters[k]
            setter(v)
        else:
            raise RuntimeError(k + ' is not a valid dynamics option')

    return options


def run(init_velocities=None, **kwargs):
    """Execute a dynamics run for the current system

    Parameters
    ----------
    init_velocities: dict
        initial velocity for each atom; 'vx', 'vy', and 'vz' each list[float]

    **kwargs: dict
        names and values from OPTIONS (see _configure function)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with index equal to step number and columns named for the
        traditional dynamics energy output entry names
    """
    natom = coor.get_natom()
    if init_velocities:
        init_vx = (ctypes.c_double * natom)(*(init_velocities['vx'][0:natom]))
        init_vy = (ctypes.c_double * natom)(*(init_velocities['vy'][0:natom]))
        init_vz = (ctypes.c_double * natom)(*(init_velocities['vz'][0:natom]))

        out_vx = (ctypes.c_double * natom)()
        out_vy = (ctypes.c_double * natom)()
        out_vz = (ctypes.c_double * natom)()
    else:
        init_vx = None
        init_vy = None
        init_vz = None

        out_vx = None
        out_vy = None
        out_vz = None

    options = _configure(**kwargs)
    success = lib.charmm.dynamics_run(ctypes.byref(options),
                                      init_vx, init_vy, init_vz,
                                      out_vx, out_vy, out_vz)

    if not init_velocities:
        out_vx = (ctypes.c_double * natom)()
        out_vy = (ctypes.c_double * natom)()
        out_vz = (ctypes.c_double * natom)()
        out_vw = (ctypes.c_double * natom)()
        natom = lib.charmm.coor_get_comparison(out_vx, out_vy, out_vz, out_vw)

    out_velocities = pandas.DataFrame({
        'vx': [out_vx[i] for i in range(natom)],
        'vy': [out_vy[i] for i in range(natom)],
        'vz': [out_vz[i] for i in range(natom)],
        'vw': [out_vw[i] for i in range(natom)]})

    return out_velocities


def get_ktable():
    """Get the ktable from the last CHARMM dynamics run

    Returns
    -------
    table : pandas.core.frame.DataFrame
             a dataframe with columns step number, time,
             some energy properties, some energy terms, and
             (for constant pressure simulations) some pressures
    """

    is_active = lib.charmm.ktable_is_active()
    if not is_active:
        return None

    nrows = lib.charmm.ktable_get_nrows()
    ncols = lib.charmm.ktable_get_ncols()
    nelts = nrows * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.ktable_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    drop_cols = [i for i in range(len(labels)) if not labels[i].strip()]
    table.drop(table.columns[drop_cols], axis=1, inplace=True)
    table['STEP'] = pandas.to_numeric(table['STEP'], downcast='unsigned')
    return table


def get_velos():
    """Get the velos from the last CHARMM dynamics run

    Returns
    -------
    table : pandas.core.frame.DataFrame
             a dataframe with columns step number, time,
             some energy properties, some energy terms, and
             (for constant pressure simulations) some pressures
    """

    is_active = lib.charmm.velos_is_active()
    if not is_active:
        return None

    nrows = lib.charmm.velos_get_nrows()
    ncols = lib.charmm.velos_get_ncols()
    nelts = nrows * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.velos_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    table['STEP'] = pandas.to_numeric(table['STEP'], downcast='unsigned')
    return table


def get_lambdata_bias():
    """Get the lambdata_bias from the last CHARMM dynamics run

    Returns
    -------
    table : pandas.core.frame.DataFrame
             a dataframe with columns step number, time,
             some energy properties, some energy terms, and
             (for constant pressure simulations) some pressures
    """

    is_active = lib.charmm.lambdata_is_active()
    if not is_active:
        return None

    nrows = lib.charmm.lambdata_bias_get_nrows()
    ncols = lib.charmm.lambdata_bias_get_ncols()
    nelts = nrows * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.lambdata_bias_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    drop_cols = [i for i in range(len(labels)) if not labels[i].strip()]
    table.drop(table.columns[drop_cols], axis=1, inplace=True)
    table['STEP'] = pandas.to_numeric(table['STEP'], downcast='unsigned')
    return table


def get_lambdata_bixlamsq():
    """Get the lambdata_bixlamsq from the last CHARMM dynamics run

    Returns
    -------
    table : pandas.core.frame.DataFrame 
             a dataframe with columns step number, time,
             some energy properties, some energy terms, and
             (for constant pressure simulations) some pressures
    """

    is_active = lib.charmm.lambdata_is_active()
    if not is_active:
        return None

    nrows = lib.charmm.lambdata_bixlamsq_get_nrows()
    ncols = lib.charmm.lambdata_bixlamsq_get_ncols()
    nelts = nrows * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.lambdata_bixlamsq_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    drop_cols = [i for i in range(len(labels)) if not labels[i].strip()]
    table.drop(table.columns[drop_cols], axis=1, inplace=True)
    table['STEP'] = pandas.to_numeric(table['STEP'], downcast='unsigned')
    return table


def get_msldata_step():
    """Get msld step data from the last step of a CHARMM dynamics run

    Returns
    -------
    table : pandas.core.frame.DataFrame
             a dataframe with columns step number, time,
             and some msld related parameters:

             NBLOCKS: number of blocks (including block 1, the environment)

             NBIASV: number of variable biases

             NSITES: number of sites (including the environment)

             TBLD: temperature for lambda dynamics

             FCNFORM: functional form for mapping between theta and lambda
    """

    is_active = lib.charmm.msldata_is_active()
    if not is_active:
        return None

    nrows = lib.charmm.msldata_get_nsteps()
    ncols = lib.charmm.msldata_step_get_ncols()
    nelts = nrows * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.msldata_step_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore') for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    table['STEP'] = pandas.to_numeric(table['STEP'], downcast='unsigned')
    table['NBIASV'] = pandas.to_numeric(table['NBIASV'], downcast='unsigned')
    table['NBLOCKS'] = pandas.to_numeric(table['NBLOCKS'], downcast='unsigned')
    table['NSITES'] = pandas.to_numeric(table['NSITES'], downcast='unsigned')
    table['FCNFORM'] = table['FCNFORM'].map(_convert_form)
    return table


def get_msldata_bias(nbiasv):
    """Get msld variable biases related data from the last step 
       of a CHARMM dynamics run.

       See CHARMM documentation 
       [block](<https://academiccharmm.org/documentation/version/c47b1/block>)
       LDBV for more information

    Parameters
    ----------
    nbiasv:  integer
          the number of variable bias rows

    Returns
    -------
    table : pandas.core.frame.DataFrame
            a dataframe with columns:

            IBVIDI: index of the first block

            IBVIDJ: index of the second block 

            IBCLAS: class of functional form

            IRREUP: REF, equilibrium distances of upper-bound biasing potentia

            IRRLOW: equilibrium distances of lower-bound biasing potential

            IKBIAS: CFORCE, force constants

            IPBIAS: NPOWER, integer power of biasing potential
    """

    is_active = lib.charmm.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.charmm.msldata_get_nsteps()
    ncols = lib.charmm.msldata_bias_get_ncols()
    nelts = nbiasv * nsteps * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size)
                   for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof,
                                                 labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.msldata_bias_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore')
                  for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    # TODO: set up the integer cols for bias data correctly
    #       template:
    #                  table['STEP'] = pandas.to_numeric(table['STEP'],
    #                                       downcast='unsigned')
    return table


def get_msldata_blocks(nblocks):
    """Get msld block data from the last step of a CHARMM dynamics run

    Parameters
    ----------
    nblocks : integer 
          the number of block rows

    Returns
    -------
    table : pandas.core.frame.DataFrame 
           a dataframe with columns:

           ISITE : site index

           BIELAM : fixed bias

           BIXLAM : lambda value
           
    """

    is_active = lib.charmm.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.charmm.msldata_get_nsteps()
    ncols = lib.charmm.msldata_block_get_ncols()
    nelts = nblocks * nsteps * ncols

    name_size = lib.charmm.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size)
                   for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof,
                                                 labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.charmm.msldata_block_get(labels_ptrs, data)

    labels_str = [label.value.decode(errors='ignore')
                  for label in labels_bufs[0:ncols]]
    labels = [label.strip() for label in labels_str]

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows, columns=labels)
    # TODO: set up the integer cols for bias data correctly
    #       template:
    #                  table['STEP'] = pandas.to_numeric(table['STEP'],
    #                                       downcast='unsigned')
    return table


def get_msldata_nsubs(nsites):
    """Get msld subsite data from the last step of a CHARMM dynamics run

    Parameters
    ----------
    nsites: integer
          the number of sites

    Returns
    -------
    table : pandas.core.frame.DataFrame
            a dataframe with nsites unnamed columns
    """

    is_active = lib.charmm.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.charmm.msldata_get_nsteps()
    ncols = nsites - 1
    nelts = nsteps * ncols
    data = (ctypes.c_double * nelts)()

    lib.charmm.msldata_nsubs_get(data)

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows)
    table.apply(pandas.to_numeric, downcast='integer', errors='ignore')
    return table


def get_msldata_thetas(nsites, nsubs=pandas.DataFrame()):
    """Get msld theta data from the last step of a CHARMM dynamics run

    Parameters
    ----------
    nsites: integer
           the number of sites
    nsubs: pandas.DataFrame
           the number of subsites for each site

    Returns
    -------
    table : pandas.core.frame.DataFrame
            a dataframe with nsites theta values and unnamed columns
    """

    is_active = lib.charmm.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.charmm.msldata_get_nsteps()

    if nsubs.empty:
        ncols = nsites - 1
    else:
        nsubs = nsubs.values.tolist()
        ncols = lib.charmm.msldata_theta_get_ncols()

    nelts = nsteps * ncols
    data = (ctypes.c_double * nelts)()

    lib.charmm.msldata_theta_get(data)

    data = data[0:nelts]
    data_rows = list()
    for i in range(0, nelts, ncols):
        data_rows.append(data[i:(i + ncols)])

    table = pandas.DataFrame(data_rows)
    return table


def get_msldata():
    """
    Get MSLD related data from the last step of a CHARMM dynamics run

    Returns
    -------
    msldata : dict
        keys are 'steps', 'biases', 'blocks', 'nsubs' (if present) and 'thetas'.

	'steps': general msld settings, eg., number of blocks, number of variable biases,
                 number of sites, temperature for lambda dynamics, functional form for
                 mapping between theta and lambda

        'biases': settings related to viariable biases

        'blocks': site index, lambda value of each substituent

        'nsubs': number of substituents at each site

        'thetas': theta of each substituent

    """
    steps = get_msldata_step()

    nbiasv = int(steps['NBIASV'].loc[steps.index[0]])
    biases = get_msldata_bias(nbiasv)

    nblocks = int(steps['NBLOCKS'].loc[steps.index[0]])
    blocks = get_msldata_blocks(nblocks)

    form = str(steps['FCNFORM'].loc[steps.index[0]])
    nsites = int(steps['NSITES'].loc[steps.index[0]])

    msldata = { 'steps': steps, 'biases': biases, 'blocks': blocks }
    if form == '2sin' or form == '2exp':
        nsubs = None
    else:
        nsubs = get_msldata_nsubs(nsites)
        msldata['nsubs'] = nsubs

    msldata['thetas'] = get_msldata_thetas(nsites, nsubs)

    return msldata


def _convert_form(form):
    new_form = 'NA'
    if form == 1.0:
        new_form = '2sin'
    elif form == 2.0:
        new_form = 'nsin'
    elif form == 3.0:
        new_form = '2exp'
    elif form == 4.0:
        new_form = 'nexp'
    elif form == 5.0:
        new_form = 'norm'
    elif form == 6.0:
        new_form = 'fixd'

    return new_form


class DynamicsScript(script.CommandScript):
    "Settings results, and methods for molecular dynamics runs"

    def run(self, append=''):
        """Run the dynamics simulation

        Parameters
        ----------
        append: str
                additional commands/options for
                the charmm command parser
        """
        if self.fill_ktable:
            lib.charmm.ktable_on()

        if self.fill_velos:
            lib.charmm.velos_on()

        if self.fill_lambdata:
            lib.charmm.lambdata_on()

        if self.fill_msldata:
            lib.charmm.msldata_on()

        super().run(append)

        if self.fill_ktable:
            self.ktable = get_ktable()
            lib.charmm.ktable_off()
            lib.charmm.ktable_del()

        if self.fill_velos:
            self.velos = get_velos()
            lib.charmm.velos_off()
            lib.charmm.velos_del()

        if self.fill_lambdata:
            self.lambdata_bias = get_lambdata_bias()
            self.lambdata_bixlamsq = get_lambdata_bixlamsq()
            lib.charmm.lambdata_off()
            lib.charmm.lambdata_del()

        if self.fill_msldata:
            self.msldata = get_msldata()
            lib.charmm.msldata_off()
            lib.charmm.msldata_del()


    def __init__(self, ktable=False, velos=False,
                 lambdata=False, msldata=False, **kwargs):
        """Class constructor

        Parameters
        ----------
        ktable: bool
                Collect final ktable data at the end of a dynamics run?
        velos: bool
                Collect final velocities at the end of a dynamics runs?
        msldata: bool
                Collect final MSLD related data at the end of a dynamics run?
        **kwargs: dict
                See CHARMM documentation 
                [dynamc](<https://academiccharmm.org/documentation/version/c47b1/dynamc>)
        """
        self.ktable = None
        self.fill_ktable = False
        if ktable:
            self.fill_ktable = True

        self.velos = None
        self.fill_velos = False
        if velos:
            self.fill_velos = True

        self.lambdata_bias = None
        self.lambdata_bixlamsq = None
        self.fill_lambdata = False
        if lambdata:
            self.fill_lambdata = True

        self.msldata = None
        self.fill_msldata = False
        if msldata:
            self.fill_msldata = True

        super().__init__('dynamics', **kwargs)
