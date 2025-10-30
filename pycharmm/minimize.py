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

"""Functions to configure and run minimization 

Corresponds to CHARMM command `MINImize`  
See [MINImize documentation](https://academiccharmm.org/documentation/version/c47b1/minimiz)  

Examples
========
>>> import pycharmm.minimize as minimize 

Following generation of PSF and building coordinates
for the system, minimization can be performed.

For OMM minimizer
>>> minimize.run_omm()

For Steepest Descent minimizer
>>> minimize.run_sd()

For ABNR minimizer
>>> minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)


"""

import ctypes

import pycharmm.lib as lib
import pycharmm.script


__all__ = ['run_abnr', 'run_omm', 'run_sd']


class MinOpts(ctypes.Structure):
    """Runtime settings for all minimization methods

    Attributes
    ----------
    nstep : int 
        number of cycles of minimization
    inbfrq : int 
        frequency of regenerating the non-bonded list
    ihbfrq : int 
        frequency of regenerating the hydrogen bond list
    nprint : int 
        step freq for printing
    gradient : int
        minimize magnitude of gradient of energy instead of energy
    numerical : int 
        forces will be determined by finite differences
    iuncrd : int 
        unit to write out a trajectory file for the minimization
    nsavc : 
        int frequency for writing out frames (only with iuncrd)
    iunxyz : int 
        unit to write out ... (?)
    nsavx : int 
        frequency for writing out frames (only with iunxyz)
    mxyz : int 
        (only with iunxyz)
    debug : int 
        extra print for debug purposes
    step : float 
        initial step size for the minimization algorithm
    tolenr : float 
        if change in total energy <= tolenr, exit
    tolgrd : float 
        if ave gradient <= tolgrd, exit
    tolstp : float 
        if ave step size <= tolstp, exit
    """
    _fields_ = [('nstep', ctypes.c_int),
                ('inbfrq', ctypes.c_int),
                ('ihbfrq', ctypes.c_int),
                ('nprint', ctypes.c_int),
                ('gradient', ctypes.c_int),
                ('numerical', ctypes.c_int),
                ('iuncrd', ctypes.c_int),
                ('nsavc', ctypes.c_int),
                ('iunxyz', ctypes.c_int),
                ('nsavx', ctypes.c_int),
                ('mxyz', ctypes.c_int),
                ('debug', ctypes.c_int),
                ('step', ctypes.c_double),
                ('tolenr', ctypes.c_double),
                ('tolgrd', ctypes.c_double),
                ('tolstp', ctypes.c_double), ]


class SDOpts(ctypes.Structure):
    """Runtime settings for steepest descent minimization method

    Attributes
    ----------
    noenergy : int 
        number of cycles of minimization
    lattice : int 
        with CRYSTAL, also optimize unit cell box size and/or shape
    nocoords : int 
        with CRYSTAL, only optimize unit cell
    """
    _fields_ = [('noenergy', ctypes.c_int),
                ('lattice', ctypes.c_int),
                ('nocoords', ctypes.c_int), ]


class AbnrOpts(ctypes.Structure):
    """runtime settings for Adopted Basis Newton-Raphson minimization

    Attributes
    ----------
    mindim : int 
        dimension of the basis set stored
    tolitr : int 
        max num of energy evals allowed for a step
    eigrng : float 
        smallest eigenval considered nonsingular
    fmem : float 
        memory factor to compute average gradient, step size
    stplim : float 
        maximum Newton Raphson step allowed
    strict : float 
        strictness of descent
    """
    _fields_ = [('mindim', ctypes.c_int),
                ('tolitr', ctypes.c_int),
                ('eigrng', ctypes.c_double),
                ('fmem', ctypes.c_double),
                ('stplim', ctypes.c_double),
                ('strict', ctypes.c_double), ]


def _configure_minimization(settings):
    """Set common minimization parameters from a dictionary of names and values

    Parameters
    ----------
    settings : dict
               a dictionary of parameters names and their desired values

    Returns
    -------
    MinOpts
              a ctypes.Structure class for options that get set when
              minimization runs
    """
    options = MinOpts(100,  # nstep
                      50,  # inbfrq
                      50,  # ihbfrq
                      10,  # nprint
                      0,  # gradient
                      0,  # numerical
                      -1,  # iuncrd
                      1,  # nsavc
                      -1,  # iunxyz
                      1,  # nsavx
                      1,  # mxyz
                      0,  # debug
                      0.02,  # step
                      0.0,  # tolenr
                      0.0,  # tolgrd
                      0.0, )  # tolstp

    for k, v in settings.items():
        # raise AttributeError if ABNR_OPTS doesn't have k field
        getattr(options, k)
        setattr(options, k, v)

    return options


def _configure_sd(settings):
    """Set steepest descent parameters from a dictionary of names and values

    Parameters
    ----------
    settings : dict
               a dictionary of parameters names and their desired values

    Returns
    -------
    MinOpts
              a ctypes.Structure class for options that get set when
              minimization runs
    """
    options = SDOpts(0,  # noenergy
                     0,  # lattice
                     0, )  # nocoords

    for k, v in settings.items():
        # raise AttributeError if ABNR_OPTS doesn't have k field
        getattr(options, k)
        setattr(options, k, v)

    return options


def _filter_attributes(obj, settings):
    """Filter settings into attributes of objs and rejects
    """
    accept = dict()
    reject = dict()
    for k, v in settings.items():
        try:
            getattr(obj, k)
            accept[k] = v
        except AttributeError:
            reject[k] = v

    return accept, reject


def run_sd(**kwargs):
    """Run steepest descent minimization

    Parameters
    ----------
    **kwargs : dict  
        settings for steepest descent minimization

    Returns
    -------
    bool
        true for success, false if there was an error
    """

    min_obj = MinOpts()
    min_settings, other_settings = _filter_attributes(min_obj, kwargs)
    min_opts = _configure_minimization(min_settings)
    sd_opts = _configure_sd(other_settings)
    status = lib.charmm.minimize_run_sd(ctypes.byref(min_opts),
                                        ctypes.byref(sd_opts))
    status = bool(status)
    return status


def _configure_abnr(settings):
    """set ABNR parameters from a dictionary of names and values

    Parameters
    ----------
    settings : dict
               a dictionary of parameters names and their desired values

    Returns
    -------
    AbnrOpts
              a ctypes.Structure class for options that get set when
              ABNR runs
    """
    options = AbnrOpts(5,  # mindim
                       100,  # tolitr
                       0.0005,  # eigrng
                       0.0,  # fmem
                       1.0,  # stplim
                       0.1, )  # strict

    for k, v in settings.items():
        # raise AttributeError if ABNR_OPTS doesn't have k field
        getattr(options, k)
        setattr(options, k, v)

    return options


def run_abnr(**kwargs):
    """Run ABNR minimization

    Parameters
    ----------
    **kwargs : dict
        settings for ABNR minimization

    Returns
    -------
    bool
        true for success, false if there was an error
    """
    min_obj = MinOpts()
    min_settings, other_settings = _filter_attributes(min_obj, kwargs)
    min_opts = _configure_minimization(min_settings)
    abnr_opts = _configure_abnr(other_settings)
    status = lib.charmm.minimize_run_abner(ctypes.byref(min_opts),
                                           ctypes.byref(abnr_opts))
    status = bool(status)
    return status


def run_omm(nstep: int, tolgrd: float, **kwargs):
    """Run OpenMM minimization

    Parameters
    ----------
    nstep : int
        number of cycles of minimization
    tolgrd : float
        if average gradient <= tolgrd, exit
    """
    min_script = pycharmm.script.CommandScript('mini omm',
                                               nstep=nstep,
                                               tolgrd=tolgrd,
                                               **kwargs)
    min_script.run()
