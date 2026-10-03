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

Integrators
===========
CHARMM provides several dynamics integrators:

- **LEAP** (default): Leapfrog Verlet - supports Langevin, CPT, SHAKE, parallel
- **ORIG**: Original Verlet - supports Langevin, TSM
- **VVER**: Velocity Verlet - supports Nose-Hoover, MTS
- **VV2**: New Velocity Verlet - supports TPCONTROL, Drude polarization
- **VER4**: 4-D Leapfrog Verlet - for energy embedding in higher dimensions
- **OMM**: OpenMM GPU-accelerated dynamics

Key Classes
===========
- `DynamicsScript` -- Main class for running dynamics simulations

Functions
=========
Temperature Control:
- `set_firstt` -- Set initial temperature (K)
- `set_finalt` -- Set final equilibrium temperature (K)
- `set_tstruc` -- Set structure equilibration temperature
- `set_teminc` -- Set temperature increment for heating
- `set_twindh` -- Set high temperature tolerance
- `set_twindl` -- Set low temperature tolerance

Time Step Control:
- `set_timest` -- Set time step (picoseconds)
- `set_akmast` -- Set time step (AKMA units)
- `set_nstep` -- Set number of dynamics steps
- `get_nstep` -- Get number of dynamics steps

Frequencies:
- `set_nprint` / `get_nprint` -- Energy print/store frequency
- `set_inbfrq` -- Nonbond list regeneration frequency
- `set_ihbfrq` -- Hydrogen bond list regeneration frequency
- `set_ilbfrq` -- Langevin region check frequency
- `set_nsavc` -- Coordinate save frequency
- `set_nsavv` -- Velocity save frequency

I/O Units:
- `set_iunwri` -- Open restart file for writing
- `set_iuncrd` -- Open coordinate trajectory for writing
- `set_iunrea` -- Open restart file for reading

Langevin Dynamics:
- `use_lang` -- Enable Langevin dynamics
- `set_fbetas` -- Set friction coefficients for atoms

Random Numbers:
- `get_nrand` -- Get number of random seeds needed
- `set_rngseeds` / `set_iseed` -- Set random number generator seeds

Other:
- `set_echeck` -- Set energy change tolerance
- `use_start` -- Use starting velocities from iasvel
- `use_restart` -- Read restart from iunrea
- `run` -- Execute dynamics run (low-level)

Data Retrieval:
- `get_ktable` -- Get energy data from dynamics run
- `get_velos` -- Get velocities from dynamics run
- `get_msldata` -- Get MSLD data from dynamics run
- `get_lambdata_bias` -- Get lambda biasing data
- `get_lambdata_bixlamsq` -- Get lambda squared data

Examples
========
A simple NVT simulation using Langevin dynamics at 298.15 K:

>>> import pycharmm
>>> import pycharmm.psf as psf
>>> import pycharmm.scalar as scalar

Set friction coefficients for Langevin
>>> n = psf.get_natom()
>>> scalar.set_fbetas([1.0] * n)

Create and run dynamics
>>> prod = pycharmm.DynamicsScript(
...     start=True, leap=True, langevin=True,
...     timestep=0.002, nstep=50000,
...     nsavc=5000, nprint=1000, iprfrq=1000,
...     isvfrq=1000, ntrfrq=5000,
...     inbfrq=-1, ihbfrq=0, imgfrq=-1,
...     iunwri=70, iuncrd=-1,
...     firstt=298.15, finalt=298.15, tbath=298.15,
...     iasors=1, iasvel=1, ichecw=0, echeck=-1
... )
>>> prod.run()

"""

import ctypes
import json
import math
import struct
import tempfile
from pathlib import Path

import numpy
import pandas

from pycharmm.loader import lib
from pycharmm.charmm_file import CharmmFile
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
    old_lang = lib.dynamics_use_lang()
    old_lang = bool(old_lang)
    return old_lang


def use_start():
    """Use starting velocities described by iasvel

    Returns
    -------
    bool
        True if start was already selected
    """
    old_start = lib.dynamics_use_start()
    old_start = bool(old_start)
    return old_start


def use_restart():
    """Dynamics is restarted by reading restart file from iunrea

    Returns
    -------
    bool
        True if start was already selected
    """
    old_restart = lib.dynamics_use_restart()
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
    old_nprint = lib.dynamics_set_nprint(ctypes.byref(new_nprint))
    return old_nprint


def get_nprint():
    """Return step freq for printing and storing energy data for dynamics runs

    Returns
    -------
    int
        the current step frequency
    """
    nprint = lib.dynamics_get_nprint()
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
    old_nstep = lib.dynamics_set_nstep(ctypes.byref(new_nstep))
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
    nstep = lib.dynamics_get_nstep()
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
    old_inbfrq = lib.dynamics_set_inbfrq(ctypes.byref(new_inbfrq))
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
    old_ihbfrq = lib.dynamics_set_ihbfrq(ctypes.byref(new_ihbfrq))
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
    old_ilbfrq = lib.dynamics_set_ilbfrq(ctypes.byref(new_ilbfrq))
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
    old_finalt = lib.dynamics_set_finalt(ctypes.byref(new_finalt))
    return old_finalt


def apply_rex_temperature_direct(expected_temperature, new_temperature):
    """Apply an accepted NVT temperature-label exchange.

    This updates CHARMM's physical thermostat targets and rescales atomic
    velocities by ``sqrt(new_temperature / expected_temperature)``. It does
    not change the effective lambda thermostat temperature. A zero lambda
    temperature sentinel may be replaced with its pre-exchange value.

    Parameters
    ----------
    expected_temperature : float
        Current physical temperature label in Kelvin.
    new_temperature : float
        Accepted physical temperature label in Kelvin.

    Returns
    -------
    bool
        True when the exchange was applied.

    Raises
    ------
    ValueError
        If either temperature is not finite and positive.
    RuntimeError
        If CHARMM is not in the expected NVT state.
    """
    expected_temperature = float(expected_temperature)
    new_temperature = float(new_temperature)
    if (not math.isfinite(expected_temperature)
            or expected_temperature <= 0.0):
        raise ValueError("expected_temperature must be finite and positive")
    if not math.isfinite(new_temperature) or new_temperature <= 0.0:
        raise ValueError("new_temperature must be finite and positive")

    expected = ctypes.c_double(expected_temperature)
    new = ctypes.c_double(new_temperature)
    status = int(lib.dynamics_exchange_temperature(
        ctypes.byref(expected), ctypes.byref(new)))
    errors = {
        0: "invalid temperature exchange input",
        -1: "temperature exchange requires ordinary NVT with one atomic bath",
        -2: "CHARMM thermostat temperature does not match expected_temperature",
        -3: "CHARMM atomic velocity state is unavailable",
        -4: "live BLaDE temperature does not match expected_temperature",
        -5: "live BLaDE temperature state is unavailable",
    }
    if status != 1:
        raise RuntimeError(errors.get(
            status, "CHARMM temperature exchange failed with status {}".format(
                status)))
    return True


def validate_rex_temperature_direct(expected_temperature):
    """Validate the current main NVT temperature without changing state."""
    return apply_rex_temperature_direct(
        expected_temperature,
        expected_temperature,
    )


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
    old_teminc = lib.dynamics_set_teminc(ctypes.byref(new_teminc))
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
    old_tstruc = lib.dynamics_set_tstruc(ctypes.byref(new_tstruc))
    return old_tstruc


def get_nrand():
    """Return the number of integers required to seed the random number generator

    the random number generator plays a role in assigning velocities

    Returns
    -------
    int
         the current number of seed integers the random number generator needs
    """
    nrand = lib.dynamics_get_nrand()
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
    new_seeds: list of int
        the list of seeds. Must be of length equal to what get_nrand() returns

    Returns
    ------
    bool
        true if the seeds were set correctly
    """
    nrand = get_nrand()
    typed_seeds = (ctypes.c_int * nrand)(*new_seeds)
    success = lib.dynamics_set_rngseeds(typed_seeds)
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
    old_timest = lib.dynamics_set_timest(ctypes.byref(new_timest))
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
    old_akmast = lib.dynamics_set_akmast(ctypes.byref(new_akmast))
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
    old_firstt = lib.dynamics_set_firstt(ctypes.byref(new_firstt))
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
    old_twindh = lib.dynamics_set_twindh(ctypes.byref(new_twindh))
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
    old_twindl = lib.dynamics_set_twindl(ctypes.byref(new_twindl))
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
    old_echeck = lib.dynamics_set_echeck(ctypes.byref(new_echeck))
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
    old_nsavc = lib.dynamics_set_nsavc(ctypes.byref(new_nsavc))
    return old_nsavc


def set_nsavv(new_nsavv):
    """Change the step freq for saving velocity data for dynamics runs

    Parameters
    ----------
    new_nsavv: int
        the new freq for saving velocity data desired

    Returns
    ------
    int
        the previous setting for the freq for saving velocity data
    """
    new_nsavv = ctypes.c_int(new_nsavv)
    old_nsavv = lib.dynamics_set_nsavv(ctypes.byref(new_nsavv))
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

    status = lib.dynamics_set_fbetas(fbetas)
    qstatus = bool(status)
    if not qstatus:
        raise RuntimeError('There was a problem setting fbetas.')

    return list(fbetas)


def _dynamics_path_ctypes(filename: str):
    """Stable path buffer for ``dynamics_set_iun*`` Fortran APIs."""
    from pycharmm.charmm_file import c_api_path_buffer

    return c_api_path_buffer(filename)


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
    fn, len_fn = _dynamics_path_ctypes(filename)

    status = lib.dynamics_set_iunwri(fn, ctypes.byref(len_fn))

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
    fn, len_fn = _dynamics_path_ctypes(filename)

    status = lib.dynamics_set_iuncrd(fn, ctypes.byref(len_fn))

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
    fn, len_fn = _dynamics_path_ctypes(filename)

    status = lib.dynamics_set_iunrea(fn, ctypes.byref(len_fn))

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


def flatten_dynamics_script(script: str) -> str:
    """Collapse a ``DynamicsScript`` string to a single ``dynopt`` keyword line."""
    body = script.replace("-\n", " ").replace("\n", " ").strip()
    if body.lower().startswith("dynamics "):
        body = body[len("dynamics ") :]
    return " ".join(body.split())


def dynamics_run_kw_available() -> bool:
    """True when ``libcharmm`` exports ``dynamics_run_kw`` (KEY_LIBRARY rebuild)."""
    return callable(getattr(lib.charmm, "dynamics_run_kw", None))


def _configure_known_only(**kwargs):
    """Apply dynamics setters/options; ignore unknown ``DynamicsScript`` keys."""
    valid_opts = dict(
        [
            ("ieqfrq", 0),
            ("ntrfrq", 100),
            ("ichecw", 0),
            ("tbath", 298.0),
            ("iasors", 0),
            ("iasvel", 1),
            ("iscale", 0),
            ("iscvel", 0),
            ("isvfrq", 100),
            ("iprfrq", 100),
            ("ihtfrq", 0),
        ]
    )

    options = OPTIONS()
    for opt, default in valid_opts.items():
        setattr(options, opt, default)

    valid_toggles = dict(
        [("lang", use_lang), ("start", use_start), ("restart", use_restart)]
    )

    valid_setters = dict(
        [
            ("nprint", set_nprint),
            ("nstep", set_nstep),
            ("inbfrq", set_inbfrq),
            ("ihbfrq", set_ihbfrq),
            ("ilbfrq", set_ilbfrq),
            ("finalt", set_finalt),
            ("teminc", set_teminc),
            ("tstruc", set_tstruc),
            ("timest", set_timest),
            ("akmast", set_akmast),
            ("firstt", set_firstt),
            ("twindh", set_twindh),
            ("twindl", set_twindl),
            ("echeck", set_echeck),
            ("nsavc", set_nsavc),
            ("nsavv", set_nsavv),
            ("iseed", set_rngseeds),
            ("fbeta", set_fbetas),
            ("iunwri", set_iunwri),
            ("iuncrd", set_iuncrd),
            ("iunrea", set_iunrea),
        ]
    )

    for k, v in kwargs.items():
        if k in valid_opts:
            setattr(options, k, v)
        elif k in valid_toggles and v:
            valid_toggles[k]()
        elif k in valid_setters:
            # Integer unit numbers are for DynamicsScript; paths are opened earlier.
            if k in ("iunwri", "iuncrd", "iunrea") and not isinstance(v, str):
                continue
            if k == "timestep":
                set_timest(float(v))
                continue
            valid_setters[k](v)

    return options


def _dynamics_velocity_ctypes_arrays(
    init_velocities: dict,
    natom: int,
) -> tuple[ctypes.Array, ctypes.Array, ctypes.Array, list]:
    """Build stable ``(c_double * natom)`` buffers for ``dynamics_run_kw``."""
    keepalive: list[ctypes.Array] = []
    out: list[ctypes.Array] = []
    for comp in ("vx", "vy", "vz"):
        vals = init_velocities[comp]
        buf = (ctypes.c_double * natom)(*(vals[i] for i in range(natom)))
        keepalive.append(buf)
        out.append(buf)
    return out[0], out[1], out[2], keepalive


def run_with_command_line(command_line: str, init_velocities=None, **kwargs):
    """Run dynamics via ``dynopt`` keyword line (KEY_LIBRARY; no ``dynamics`` script)."""
    if not dynamics_run_kw_available():
        raise RuntimeError(
            "dynamics_run_kw is not exported by libcharmm; rebuild with api_dynamics.F90"
        )

    from pycharmm.charmm_file import c_api_string_buffer

    natom = coor.get_natom()
    vel_keepalive: list[ctypes.Array] = []
    if init_velocities is not None:
        init_vx, init_vy, init_vz, vel_keepalive = _dynamics_velocity_ctypes_arrays(
            init_velocities, natom
        )
        out_vx = (ctypes.c_double * natom)()
        out_vy = (ctypes.c_double * natom)()
        out_vz = (ctypes.c_double * natom)()
    else:
        init_vx = init_vy = init_vz = None
        out_vx = out_vy = out_vz = None

    options = _configure_known_only(**kwargs)
    buf, buflen = c_api_string_buffer(command_line)
    fn = lib.dynamics_run_kw

    # Fortran bind(c) ABI requirements for dynamics_run_kw:
    #
    # 1. c_kw_len is declared `integer(c_int), value` → pass by VALUE not byref.
    #    ctypes byref() would pass a pointer-to-int instead of an int, corrupting
    #    the argument.
    #
    # 2. in_vx/vy/vz/out_vx/vy/vz are `dimension(:), optional` in bind(c).
    #    gfortran's CFI ABI signals absent optional arrays via a NULL CFI-descriptor
    #    pointer.  Simply omitting these arguments leaves garbage in stack/register
    #    positions that _gfortran_cfi_desc_to_gfc_desc tries to dereference
    #    (→ segfault at address 0x6 or similar small offset).
    #    We must always pass all 9 arguments; absent ones must be NULL (c_void_p).
    #
    # 3. For present arrays, gfortran still expects a CFI descriptor (not a raw
    #    data pointer).  Passing raw ctypes Arrays as c_void_p gives gfortran a
    #    non-CFI pointer and will likely segfault inside _gfortran_cfi_desc_to_gfc_desc.
    #    The velocity-injection path (MMML_BUSSI_INIT_VELOCITIES_HANDOFF=1) is
    #    therefore kept explicitly broken by design until a proper CFI wrapper exists.

    fn.argtypes = [
        ctypes.POINTER(OPTIONS),  # options (by reference — struct)
        ctypes.c_char_p,          # c_kw   (char * — assumed-size character)
        ctypes.c_int,             # c_kw_len (VALUE — must NOT be byref)
        ctypes.c_void_p,          # in_vx  (CFI descriptor ptr or NULL)
        ctypes.c_void_p,          # in_vy
        ctypes.c_void_p,          # in_vz
        ctypes.c_void_p,          # out_vx
        ctypes.c_void_p,          # out_vy
        ctypes.c_void_p,          # out_vz
    ]
    fn.restype = ctypes.c_int

    if init_velocities is not None:
        # Velocity path: pass raw data pointers. NOTE: gfortran will interpret
        # these as CFI descriptors and will likely crash (see note 3 above).
        # This path is only reached when MMML_BUSSI_INIT_VELOCITIES_HANDOFF=1.
        success = fn(
            ctypes.byref(options),
            buf,
            buflen,  # c_int value — NOT byref
            ctypes.cast(init_vx, ctypes.c_void_p),
            ctypes.cast(init_vy, ctypes.c_void_p),
            ctypes.cast(init_vz, ctypes.c_void_p),
            ctypes.cast(out_vx, ctypes.c_void_p),
            ctypes.cast(out_vy, ctypes.c_void_p),
            ctypes.cast(out_vz, ctypes.c_void_p),
        )
    else:
        # No-velocity path: pass NULL for all 6 optional array positions.
        # Fortran present() returns .false. for NULL CFI descriptor pointers,
        # so dynopt takes the non-velocity branch (correct behaviour).
        success = fn(
            ctypes.byref(options),
            buf,
            buflen,  # c_int value — NOT byref
            None,  # in_vx  → absent → NULL CFI desc → present()=.false.
            None,  # in_vy
            None,  # in_vz
            None,  # out_vx
            None,  # out_vy
            None,  # out_vz
        )



    if not success:
        raise RuntimeError("dynamics_run_kw failed")

    if not init_velocities:
        out_vx = (ctypes.c_double * natom)()
        out_vy = (ctypes.c_double * natom)()
        out_vz = (ctypes.c_double * natom)()
        out_vw = (ctypes.c_double * natom)()
        lib.coor_get_comparison(out_vx, out_vy, out_vz, out_vw)
    else:
        out_vw = (ctypes.c_double * natom)()

    return pandas.DataFrame(
        {
            "vx": [out_vx[i] for i in range(natom)],
            "vy": [out_vy[i] for i in range(natom)],
            "vz": [out_vz[i] for i in range(natom)],
            "vw": [out_vw[i] for i in range(natom)],
        }
    )


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
    lib.dynamics_run(ctypes.byref(options),
                     init_vx, init_vy, init_vz,
                     out_vx, out_vy, out_vz)

    if not init_velocities:
        out_vx = (ctypes.c_double * natom)()
        out_vy = (ctypes.c_double * natom)()
        out_vz = (ctypes.c_double * natom)()
        out_vw = (ctypes.c_double * natom)()
        natom = lib.coor_get_comparison(out_vx, out_vy, out_vz, out_vw)

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

    is_active = lib.ktable_is_active()
    if not is_active:
        return None

    nrows = lib.ktable_get_nrows()
    ncols = lib.ktable_get_ncols()
    nelts = nrows * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.ktable_get(labels_ptrs, data)

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

    is_active = lib.velos_is_active()
    if not is_active:
        return None

    nrows = lib.velos_get_nrows()
    ncols = lib.velos_get_ncols()
    nelts = nrows * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.velos_get(labels_ptrs, data)

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

    is_active = lib.lambdata_is_active()
    if not is_active:
        return None

    nrows = lib.lambdata_bias_get_nrows()
    ncols = lib.lambdata_bias_get_ncols()
    nelts = nrows * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.lambdata_bias_get(labels_ptrs, data)

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

    is_active = lib.lambdata_is_active()
    if not is_active:
        return None

    nrows = lib.lambdata_bixlamsq_get_nrows()
    ncols = lib.lambdata_bixlamsq_get_ncols()
    nelts = nrows * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.lambdata_bixlamsq_get(labels_ptrs, data)

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


def _read_legacy_lmd_record(raw, offset):
    if offset + 8 > len(raw):
        raise ValueError("truncated legacy lmd record marker")
    record_size = struct.unpack_from("<i", raw, offset)[0]
    record_start = offset + 4
    record_end = record_start + record_size
    if record_size < 0 or record_end + 4 > len(raw):
        raise ValueError("truncated legacy lmd record payload")
    trailer_size = struct.unpack_from("<i", raw, record_end)[0]
    if trailer_size != record_size:
        raise ValueError("legacy lmd record marker mismatch")
    return memoryview(raw)[record_start:record_end], record_end + 4


def _lambda_parquet_frame(lmd_path):
    """Read one native MSLD dynamics output into wide Parquet columns."""
    raw = Path(lmd_path).read_bytes()

    header_record, offset = _read_legacy_lmd_record(raw, 0)
    if len(header_record) != 84:
        raise ValueError("invalid MSLD lambda header size")
    header = numpy.frombuffer(
        header_record,
        dtype=[("hdr", "S4"), ("icntrl", "<i4", 20)],
        count=1,
    )[0]
    if header["hdr"] != b"MSLD":
        raise ValueError("lambda_parquet supports MSLD lambda files only")
    icntrl = header["icntrl"]

    delta_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(delta_record) != 4:
        raise ValueError("invalid MSLD lambda timestep record")
    delta_t = (
        float(numpy.frombuffer(delta_record, dtype="<f4", count=1)[0])
        * 4.88882129e-2
    )

    _title_record, offset = _read_legacy_lmd_record(raw, offset)

    nblocks = int(icntrl[6])
    if nblocks < 1:
        raise ValueError("legacy lmd header has no lambda blocks")
    nsites = int(icntrl[10])
    if nsites < 1 or nsites > nblocks:
        raise ValueError("legacy lmd header has an invalid MSLD site count")

    nbias_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(nbias_record) != 4:
        raise ValueError("invalid MSLD lambda bias-count record")
    nbias = int(numpy.frombuffer(nbias_record, dtype="<i4", count=1)[0])
    if nbias < 0:
        raise ValueError("invalid MSLD lambda bias count")
    bias_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(bias_record) != 7 * nbias * 4:
        raise ValueError("invalid MSLD lambda bias record")
    site_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(site_record) != nblocks * 4:
        raise ValueError("invalid MSLD lambda site record")
    temperature_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(temperature_record) != 4:
        raise ValueError("invalid MSLD lambda temperature record")
    lambda_temperature = float(
        numpy.frombuffer(temperature_record, dtype="<f4", count=1)[0]
    )
    bielam_record, offset = _read_legacy_lmd_record(raw, offset)
    if len(bielam_record) != nblocks * 4:
        raise ValueError("invalid MSLD lambda BIELAM record")
    bielam = numpy.frombuffer(
        bielam_record,
        dtype="<f4",
        count=nblocks,
    ).tolist()

    theta_count = nsites - 1 if int(icntrl[11]) == 2 else nblocks - 1
    lambda_record_bytes = nblocks * 4 + 8
    theta_record_bytes = theta_count * 4 + 8
    frame_bytes = lambda_record_bytes + theta_record_bytes
    payload = memoryview(raw)[offset:]
    if len(payload) % frame_bytes != 0:
        raise ValueError("legacy lmd payload does not align to whole frames")

    nframes = len(payload) // frame_bytes
    columns = ["LAM{:02d}".format(index) for index in range(1, nblocks)]
    if nframes == 0:
        frame = pandas.DataFrame(
            {"time": numpy.empty(0, dtype=numpy.float64)}
        )
        for column in columns:
            frame[column] = numpy.empty(0, dtype=numpy.float32)
        frame.attrs.update(
            lambda_temperature=lambda_temperature,
            bielam=bielam,
        )
        return frame

    frame_words = numpy.frombuffer(payload, dtype="<i4").reshape(nframes, -1)
    if (
        numpy.any(frame_words[:, 0] != nblocks * 4)
        or numpy.any(frame_words[:, nblocks + 1] != nblocks * 4)
        or numpy.any(frame_words[:, nblocks + 2] != theta_count * 4)
        or numpy.any(frame_words[:, -1] != theta_count * 4)
    ):
        raise ValueError("legacy lmd frame record marker mismatch")

    lambda_values = numpy.ndarray(
        shape=(nframes, nblocks - 1),
        dtype="<f4",
        buffer=payload,
        offset=8,
        strides=(frame_bytes, 4),
    ).copy()

    start_time = int(icntrl[1]) * delta_t
    time_step = int(icntrl[2]) * delta_t
    timestamps = (
        start_time
        + numpy.arange(nframes, dtype=numpy.float64) * time_step
    )
    frame = pandas.DataFrame(lambda_values, columns=columns)
    frame.insert(0, "time", timestamps)
    frame.attrs.update(
        lambda_temperature=lambda_temperature,
        bielam=bielam,
    )
    return frame


def _preflight_lambda_parquet(filepath, compression):
    """Validate Parquet support and destination before dynamics starts."""
    try:
        import pyarrow
    except ImportError as exc:
        raise RuntimeError("writing lambda parquet requires pyarrow") from exc

    try:
        codec_available = pyarrow.Codec.is_available(compression)
    except ValueError:
        codec_available = False
    if not codec_available:
        raise ValueError(
            "unsupported lambda parquet compression codec: {}".format(
                compression
            )
        )

    filepath = Path(filepath)
    if filepath.is_dir():
        raise IsADirectoryError(
            "lambda parquet destination is a directory: {}".format(filepath)
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".{}_".format(filepath.name),
        suffix=".probe",
        dir=filepath.parent,
    ):
        pass


def _lambda_parquet_path_for_rank(filepath):
    """Resolve a collision-free Parquet path for the current MPI rank."""
    from pycharmm.replica_exchange import _default_comm

    comm = _default_comm()
    rank = int(comm.Get_rank())
    path = str(filepath)
    if int(comm.Get_size()) > 1 and "{rank}" not in path:
        raise ValueError(
            "lambda_parquet requires a {rank} placeholder with multiple "
            "MPI ranks"
        )
    return Path(path.replace("{rank}", str(rank)))


def _write_lambda_parquet(
    lmd_path,
    filepath,
    compression="snappy",
    ph=None,
    mpi_rank=None,
):
    """Convert a native MSLD lambda trajectory atomically to Parquet."""
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as exc:
        raise RuntimeError("writing lambda parquet requires pyarrow") from exc

    frame = _lambda_parquet_frame(lmd_path)
    if ph is not None:
        frame.attrs["ph"] = float(ph)
    if mpi_rank is not None:
        frame.attrs["mpi_rank"] = int(mpi_rank)
    attrs_json = json.dumps(
        frame.attrs,
        allow_nan=False,
        separators=(",", ":"),
    )
    metadata = {
        b"PANDAS_ATTRS": attrs_json.encode(),
        b"charmm.lambda_temperature": str(
            frame.attrs["lambda_temperature"]
        ).encode(),
        b"charmm.bielam": json.dumps(
            frame.attrs["bielam"],
            allow_nan=False,
            separators=(",", ":"),
        ).encode(),
    }
    if ph is not None:
        metadata[b"charmm.ph"] = str(frame.attrs["ph"]).encode()
    if mpi_rank is not None:
        metadata[b"charmm.mpi_rank"] = str(frame.attrs["mpi_rank"]).encode()
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    metadata = {**(table.schema.metadata or {}), **metadata}
    table = table.replace_schema_metadata(metadata)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=".{}_".format(filepath.name),
        suffix=".parquet",
        dir=filepath.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        pyarrow.parquet.write_table(
            table,
            temporary_path,
            compression=compression,
        )
        temporary_path.replace(filepath)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return filepath


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

    is_active = lib.msldata_is_active()
    if not is_active:
        return None

    nrows = lib.msldata_get_nsteps()
    ncols = lib.msldata_step_get_ncols()
    nelts = nrows * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size) for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof, labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.msldata_step_get(labels_ptrs, data)

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

    is_active = lib.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.msldata_get_nsteps()
    ncols = lib.msldata_bias_get_ncols()
    nelts = nbiasv * nsteps * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size)
                   for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof,
                                                 labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.msldata_bias_get(labels_ptrs, data)

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

    is_active = lib.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.msldata_get_nsteps()
    ncols = lib.msldata_block_get_ncols()
    nelts = nblocks * nsteps * ncols

    name_size = lib.dataframe_get_name_size()
    labels_bufs = [ctypes.create_string_buffer(name_size)
                   for _ in range(ncols)]
    labels_ptrs = (ctypes.c_char_p * ncols)(*map(ctypes.addressof,
                                                 labels_bufs))

    data = (ctypes.c_double * nelts)()

    lib.msldata_block_get(labels_ptrs, data)

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

    is_active = lib.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.msldata_get_nsteps()
    ncols = nsites - 1
    nelts = nsteps * ncols
    data = (ctypes.c_double * nelts)()

    lib.msldata_nsubs_get(data)

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

    is_active = lib.msldata_is_active()
    if not is_active:
        return None

    nsteps = lib.msldata_get_nsteps()

    if nsubs.empty:
        ncols = nsites - 1
    else:
        nsubs = nsubs.values.tolist()
        ncols = lib.msldata_theta_get_ncols()

    nelts = nsteps * ncols
    data = (ctypes.c_double * nelts)()

    lib.msldata_theta_get(data)

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
    """Settings, results, and methods for molecular dynamics runs.

    This class provides a comprehensive interface to CHARMM dynamics,
    supporting multiple integrators, thermostats, and barostats.

    Parameters (passed to __init__)
    --------------------------------
    ktable : bool
        Collect energy data at end of run. (default: False)
    velos : bool
        Collect velocities at end of run. (default: False)
    lambdata : bool
        Collect lambda dynamics data. (default: False)
    lambda_parquet : str or path-like, optional
        Write MSLD ``time + lambda`` columns as Parquet. Positive ``nstep``
        and ``nsavl`` dynamics options are required. One file is written per
        dynamics run; use a distinct path for each replica-exchange segment.
        MPI runs must include ``{rank}`` in the path.
    msldata : bool
        Collect MSLD data at end of run. (default: False)

    Integrator Selection (kwargs):
    - leap : bool - Use Leapfrog Verlet (default)
    - orig : bool - Use Original Verlet
    - vver : bool - Use Velocity Verlet
    - vv2 : bool - Use New Velocity Verlet
    - ver4 : bool - Use 4-D Verlet
    - omm : bool - Use OpenMM GPU dynamics

    Dynamics Type (kwargs):
    - start : bool - Start new dynamics (default)
    - restart : bool - Restart from file
    - langevin : bool - Use Langevin thermostat
    - cpt : bool - Use CPT (constant pressure/temperature)
    - nose : bool - Use Nose-Hoover thermostat (with vver)

    Time Parameters (kwargs):
    - timestep : float - Time step in ps (default: 0.001)
    - nstep : int - Number of steps (default: 100)

    Frequencies (kwargs):
    - nprint : int - Energy print frequency (default: 10)
    - nsavc : int - Coordinate save frequency (default: 10)
    - nsavv : int - Velocity save frequency (default: 10)
    - nsavl : int - Lambda histogram save frequency
    - iprfrq : int - Average/RMS calculation frequency (default: 100)
    - isvfrq : int - Restart file write frequency
    - ntrfrq : int - Translation/rotation removal frequency
    - inbfrq : int - Nonbond list update frequency (default: 50, -1=auto)
    - ihbfrq : int - Hydrogen bond list update frequency (default: 50)
    - ilbfrq : int - Langevin region check frequency (default: 50)
    - imgfrq : int - Image update frequency (for IMAGES/CRYSTAL)
    - ihtfrq : int - Heating frequency
    - ieqfrq : int - Equilibration velocity assignment frequency

    I/O Units (kwargs):
    - iunrea : int - Unit for reading restart (-1=none)
    - iunwri : int - Unit for writing restart (-1=none)
    - iuncrd : int - Unit for coordinate trajectory (-1=none)
    - iunvel : int - Unit for velocity trajectory (-1=none)
    - iunldm : int - Unit for lambda histograms (-1=none)
    - kunit : int - Unit for energy output (-1=none)

    Temperature (kwargs):
    - firstt : float - Initial temperature (K)
    - finalt : float - Final temperature (K) (default: 298.0)
    - tbath : float - Bath temperature for Langevin (K)
    - tstruct : float - Structure equilibration temperature
    - teminc : float - Temperature increment for heating (default: 5.0)
    - twindh : float - High temperature window tolerance
    - twindl : float - Low temperature window tolerance

    Velocity Control (kwargs):
    - iasvel : int - Velocity assignment mode (1=Gaussian)
    - iasors : int - Assign (0) or scale (1) velocities
    - iscale : int - Scaling option
    - iscvel : int - Velocity scaling mode
    - iseed : list[int] - Random number seeds
    - scale : float - Velocity scaling factor
    - ichecw : int - Check temperature window (0=no, 1=yes)

    Other Options (kwargs):
    - echeck : float - Energy change tolerance per step
    - rbuffer : float - Langevin region buffer distance
    - ndeg : int - Number of degrees of freedom
    - tol : float - SHAKE tolerance

    Attributes
    ----------
    ktable : pandas.DataFrame or None
        Energy data from dynamics run (if ktable=True)
    velos : pandas.DataFrame or None
        Velocity data from dynamics run (if velos=True)
    msldata : dict or None
        MSLD data from dynamics run (if msldata=True)
    lambdata_bias : pandas.DataFrame or None
        Lambda bias data (if lambdata=True)
    lambdata_bixlamsq : pandas.DataFrame or None
        Lambda squared data (if lambdata=True)
    lambda_parquet_path : pathlib.Path or None
        Requested Parquet path or MPI ``{rank}`` path template

    Examples
    --------
    >>> import pycharmm

    NVT Langevin dynamics
    >>> dyn = pycharmm.DynamicsScript(
    ...     start=True, leap=True, langevin=True,
    ...     timestep=0.002, nstep=10000,
    ...     firstt=300.0, finalt=300.0, tbath=300.0,
    ...     inbfrq=-1, imgfrq=50
    ... )
    >>> dyn.run()

    NPT dynamics with CPT
    >>> dyn = pycharmm.DynamicsScript(
    ...     start=True, leap=True, cpt=True,
    ...     timestep=0.002, nstep=10000,
    ...     pcons=True, pmass=500, pgamma=20.0,
    ...     hoession=True, pconsttype='berendsen'
    ... )
    >>> dyn.run()

    Collect energy data
    >>> dyn = pycharmm.DynamicsScript(
    ...     ktable=True, nstep=1000, nprint=10
    ... )
    >>> dyn.run()
    >>> print(dyn.ktable)  # pandas DataFrame with energy vs time
    """

    def run(self, append=''):
        """Run the dynamics simulation.

        Parameters
        ----------
        append : str
            Additional commands/options for the CHARMM command parser.
            (default: '')

        Returns
        -------
        bool
            True if dynamics completed normally, False if interrupted
            (only relevant when using BLaDE engine with handle_interrupt=True).
        """
        import warnings
        import pycharmm.restraints as restraints

        import pycharmm.domdec as domdec
        if self._engine == 'domdec' or domdec.is_enabled():
            domdec._check_crystal_compatibility()
        if (self.lambda_parquet_path is not None
                and not self._lambda_parquet_iunldm_is_managed()):
            raise ValueError(
                "lambda_parquet with an explicit iunldm is unsupported; leave "
                "iunldm unset or set it to -1 so pyCHARMM can manage the "
                "lambda file and preserve the parquet layout"
            )
        if self.lambda_parquet_path is not None:
            from pycharmm.replica_exchange import _default_comm

            self._validate_lambda_parquet_options()
            lambda_parquet_path = _lambda_parquet_path_for_rank(
                self.lambda_parquet_path
            )
            lambda_parquet_rank = int(_default_comm().Get_rank())
            _preflight_lambda_parquet(
                lambda_parquet_path,
                self.lambda_parquet_compression,
            )
        else:
            lambda_parquet_path = None
            lambda_parquet_rank = None

        # Check restraint compatibility with selected engine
        if self._warn_restraints and self._engine != 'standard':
            conflicts = restraints._state._check_backend_conflicts(self._engine)
            if conflicts:
                engine_name = {'openmm': 'OpenMM', 'blade': 'BLaDE',
                               'domdec': 'DOMDEC'}.get(self._engine, self._engine)
                warnings.warn(
                    f"Active restraints incompatible with {engine_name}: "
                    f"{', '.join(conflicts)}. "
                    f"These restraints will not be evaluated correctly.",
                    UserWarning, stacklevel=2
                )
            # Update backend state in restraints module
            try:
                restraints.set_backend(self._engine)
            except restraints.IncompatibleBackendError:
                pass  # Warning already issued above

        # Install signal handler for BLaDE engine for graceful Ctrl+C handling
        interrupted = False
        if self._engine == 'blade' and self._handle_interrupt:
            import pycharmm.blade as blade
            blade.install_signal_handler()

        lambda_file = None
        lambda_path = None
        retain_lambda_path = False
        previous_iunldm = self.opts.get('iunldm')
        try:
            if self.lambda_parquet_path is not None:
                temp_file = tempfile.NamedTemporaryFile(
                    prefix='pycharmm_lambda_',
                    suffix='.lmd',
                    delete=False,
                )
                temp_file.close()
                lambda_path = Path(temp_file.name)
                retain_lambda_path = True
                lambda_file = CharmmFile(
                    file_name=str(lambda_path),
                    file_unit=-1,
                    read_only=False,
                    formatted=False,
                )
                if not lambda_file.is_open:
                    raise OSError(
                        "failed to open temporary MSLD lambda file: "
                        "{}".format(lambda_path)
                    )
                replica_paths = list(
                    lambda_path.parent.glob(lambda_path.name + "_*")
                )
                if len(replica_paths) == 1:
                    lambda_path.unlink()
                    lambda_path = replica_paths[0]
                self.opts['iunldm'] = (
                    'iunldm {} -\n'.format(lambda_file.file_unit)
                )

            if self.fill_ktable:
                lib.ktable_on()

            if self.fill_velos:
                lib.velos_on()

            if self.fill_lambdata:
                lib.lambdata_on()

            if self.fill_msldata:
                lib.msldata_on()

            super().run(append)

            if lambda_file is not None:
                if not lambda_file.close():
                    raise OSError(
                        "failed to close temporary MSLD lambda file: "
                        "{}".format(lambda_path)
                    )
                lambda_file = None

            # Check before conversion so an incomplete interrupted frame can
            # be retained without turning Ctrl+C into a conversion exception.
            if self._engine == 'blade' and self._handle_interrupt:
                import pycharmm.blade as blade
                interrupted = blade.check_interrupt()

            if self.fill_ktable:
                self.ktable = get_ktable()
                lib.ktable_off()
                lib.ktable_del()

            if self.fill_velos:
                self.velos = get_velos()
                lib.velos_off()
                lib.velos_del()

            if self.fill_lambdata:
                self.lambdata_bias = get_lambdata_bias()
                self.lambdata_bixlamsq = get_lambdata_bixlamsq()
                lib.lambdata_off()
                lib.lambdata_del()

            if self.fill_msldata:
                self.msldata = get_msldata()
                lib.msldata_off()
                lib.msldata_del()

            if lambda_parquet_path is not None:
                try:
                    _write_lambda_parquet(
                        lambda_path,
                        lambda_parquet_path,
                        compression=self.lambda_parquet_compression,
                        ph=self.lambda_parquet_ph,
                        mpi_rank=lambda_parquet_rank,
                    )
                    retain_lambda_path = False
                except Exception as exc:
                    if self._engine == 'blade' and self._handle_interrupt:
                        import pycharmm.blade as blade
                        interrupted = blade.check_interrupt()
                    if not interrupted:
                        raise RuntimeError(
                            "lambda parquet conversion failed; native MSLD "
                            "lambda data retained at {}".format(lambda_path)
                        ) from exc

        finally:
            # Always restore signal handler for BLaDE
            if self._engine == 'blade' and self._handle_interrupt:
                import pycharmm.blade as blade
                blade.restore_signal_handler()
                interrupted = interrupted or blade.check_interrupt()
                blade.set_interrupt(0)  # Clear flag for next run
            if lambda_file is not None:
                lambda_file.close()
            if lambda_path is not None:
                if previous_iunldm is None:
                    self.opts.pop('iunldm', None)
                else:
                    self.opts['iunldm'] = previous_iunldm
                if (
                    retain_lambda_path
                    and lambda_path.exists()
                    and lambda_path.stat().st_size
                ):
                    warnings.warn(
                        "lambda parquet was not written; native MSLD lambda "
                        "data retained at {}".format(lambda_path),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    try:
                        lambda_path.unlink()
                    except FileNotFoundError:
                        pass

        return not interrupted


    def __init__(self, ktable=False, velos=False, lambdata=False,
                 msldata=False, warn_restraints=True, handle_interrupt=True,
                 *, lambda_parquet=None,
                 lambda_parquet_compression='snappy',
                 lambda_parquet_ph=None, **kwargs):
        """Class constructor

        Parameters
        ----------
        ktable: bool
                Collect final ktable data at the end of a dynamics run?
        velos: bool
                Collect final velocities at the end of a dynamics runs?
        msldata: bool
                Collect final MSLD related data at the end of a dynamics run?
        warn_restraints: bool
                If True (default), warn if active restraints are incompatible
                with the selected engine (omm, blade, domdec).
        handle_interrupt: bool
                If True (default), install a signal handler so Ctrl+C stops
                BLaDE dynamics gracefully instead of terminating Python.
                Only applies when using blade=True.
        lambda_parquet: str or path-like, optional
                Write MSLD lambda/time columns as Parquet without enabling
                CHARMM's in-memory lambdata tables. Requires positive nstep
                and nsavl dynamics options. Existing files are replaced, so
                segmented runs must use a distinct path for each segment.
                MPI runs must include ``{rank}`` in the path.
        lambda_parquet_compression: str
                Parquet compression codec. (default: snappy)
        lambda_parquet_ph: float, optional
                pH label stored with this Parquet segment. Set it explicitly
                for each pH replica-exchange segment. The value must match
                the active CHARMM pH/MSLD state.
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
        self.lambda_parquet_path = None
        self.lambda_parquet_compression = str(lambda_parquet_compression)
        self.lambda_parquet_ph = None
        if lambda_parquet_ph is not None:
            self.lambda_parquet_ph = float(lambda_parquet_ph)
            if not numpy.isfinite(self.lambda_parquet_ph):
                raise ValueError("lambda_parquet_ph must be finite")
        if lambda_parquet is not None:
            self.lambda_parquet_path = Path(lambda_parquet)
        if lambdata:
            self.fill_lambdata = True

        self.msldata = None
        self.fill_msldata = False
        if msldata:
            self.fill_msldata = True

        # Check restraint compatibility with selected engine
        self._engine = 'standard'
        self._handle_interrupt = handle_interrupt
        if kwargs.get('omm', False):
            self._engine = 'openmm'
        elif kwargs.get('blade', False):
            self._engine = 'blade'
        elif kwargs.get('domdec', False):
            self._engine = 'domdec'

        self._warn_restraints = warn_restraints

        super().__init__('dynamics', **kwargs)

    def _lambda_parquet_iunldm_is_managed(self):
        iunldm = self.opts.get('iunldm')
        if iunldm is None:
            return True

        fields = str(iunldm).split()
        if len(fields) < 2 or fields[0].lower() != 'iunldm':
            return False
        try:
            return int(fields[1]) == -1
        except ValueError:
            return False

    def _validate_lambda_parquet_options(self):
        nstep = self._integer_option('nstep')
        nsavl = self._integer_option('nsavl')
        if nstep is None or nstep < 1:
            raise ValueError("lambda_parquet requires a positive nstep")
        if nsavl is None or nsavl < 1 or nsavl > nstep:
            raise ValueError(
                "lambda_parquet requires nsavl between 1 and nstep"
            )
        if self.lambda_parquet_ph is not None:
            import pycharmm.block as block

            active_ph = block.get_ph_direct()
            if active_ph is None:
                raise ValueError(
                    "lambda_parquet_ph requires active pH/MSLD state"
                )
            if not numpy.isclose(
                active_ph,
                self.lambda_parquet_ph,
                rtol=1.0e-7,
                atol=1.0e-7,
            ):
                raise ValueError(
                    "lambda_parquet_ph does not match the active pH/MSLD state"
                )

    def _integer_option(self, name):
        fields = str(self.opts.get(name, '')).split()
        if len(fields) < 2 or fields[0].lower() != name:
            return None
        try:
            return int(fields[1])
        except ValueError:
            return None
