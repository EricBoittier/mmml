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

Available Minimization Methods
==============================
- **SD** (Steepest Descent): Simple, robust for poor conformations
- **CONJ** (Conjugate Gradient): Better convergence than SD
- **ABNR** (Adopted Basis Newton-Raphson): Best for most circumstances
- **NRAP** (Newton-Raphson): Fast convergence but O(n²) storage
- **POWELL** (Conjugate Gradient Powell): Improved efficiency over CONJ
- **TN** (Truncated Newton): Competitive with ABNR, uses preconditioning
- **OMM** (OpenMM): Simple local minimizer through OpenMM interface

Examples
========
>>> import pycharmm.minimize as minimize

Following generation of PSF and building coordinates
for the system, minimization can be performed.

For OMM minimizer
>>> minimize.run_omm(nstep=1000, tolgrd=1e-3)

For Steepest Descent minimizer
>>> minimize.run_sd(nstep=500)

For ABNR minimizer
>>> minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)

For Conjugate Gradient minimizer
>>> minimize.run_conj(nstep=500, ncgcyc=100)

For Powell minimizer
>>> minimize.run_powell(nstep=500)

For Newton-Raphson minimizer (small systems only)
>>> minimize.run_nrap(nstep=100, tfreq=1.0)

For Truncated Newton minimizer
>>> minimize.run_tn(nstep=500, tolgrd=1e-6)


"""

import ctypes

# import pycharmm.loader as lib
from pycharmm.loader import lib
import pycharmm.script


__all__ = ['run_abnr', 'run_blade', 'run_conj', 'run_nrap', 'run_omm',
           'run_powell', 'run_sd', 'run_tn']


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
    status = lib.minimize_run_sd(ctypes.byref(min_opts),
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
    lattice = bool(kwargs.pop("lattice", False))
    nocoords = bool(kwargs.pop("nocoords", False))
    min_obj = MinOpts()
    min_settings, other_settings = _filter_attributes(min_obj, kwargs)
    min_opts = _configure_minimization(min_settings)
    abnr_opts = _configure_abnr(other_settings)
    lat_fn = getattr(lib, "minimize_run_abnr_lattice", None)
    if (lattice or nocoords) and callable(lat_fn):
        lat = ctypes.c_int(1 if lattice else 0)
        noco = ctypes.c_int(1 if nocoords else 0)
        status = lat_fn(
            ctypes.byref(min_opts),
            ctypes.byref(abnr_opts),
            lat,
            noco,
        )
    else:
        if lattice or nocoords:
            return run_sd(
                lattice=lattice,
                nocoords=nocoords,
                **min_settings,
            )
        status = lib.minimize_run_abner(
            ctypes.byref(min_opts),
            ctypes.byref(abnr_opts),
        )
    status = bool(status)
    return status


def run_omm(nstep: int = 0, tolgrd: float = 0.01,
            warn_restraints: bool = True, **kwargs):
    """Run OpenMM minimization

    Uses OpenMM's LocalEnergyMinimizer for simple local minimization.

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. If 0, minimization proceeds
        until RMS gradient reaches tolgrd. (default: 0)
    tolgrd : float
        RMS gradient tolerance for convergence. (default: 0.01)
    warn_restraints : bool, optional
        If True (default), warn if incompatible restraints are active.
    **kwargs : dict
        Additional minimization settings.

    Notes
    -----
    This function only uses NSTEP and TOLGRD and does not report energy,
    but returns the locally minimized structure.

    OpenMM does NOT support the following restraint types:
    - NOE/PNOE restraints
    - RESD (restrained distances)
    - IC (internal coordinate) restraints
    - DROPLET restraints

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_omm(nstep=1000, tolgrd=1e-3)
    >>> minimize.run_omm(tolgrd=1e-4)  # Run until convergence
    """
    import warnings
    import pycharmm.restraints as restraints

    # Check for incompatible restraints
    if warn_restraints:
        conflicts = restraints._state._check_backend_conflicts('openmm')
        if conflicts:
            warnings.warn(
                f"Active restraints incompatible with OpenMM: {', '.join(conflicts)}. "
                f"These restraints will not be evaluated correctly in OpenMM minimization.",
                UserWarning, stacklevel=2
            )
        # Update backend state
        restraints.set_backend('openmm')

    min_script = pycharmm.script.CommandScript('mini omm',
                                               nstep=nstep,
                                               tolgrd=tolgrd,
                                               **kwargs)
    min_script.run()


def run_blade(nstep: int = 100, tolgrd: float = 1e-3,
              method: str = 'sd',
              warn_restraints: bool = True,
              handle_interrupt: bool = True, **kwargs):
    """Run BLaDE GPU-accelerated minimization

    Uses BLaDE (Basic LAmbda Dynamics Engine) for
    GPU-accelerated energy and minimization.

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. (default: 100)
    tolgrd : float
        Accepted for compatibility; BLaDE does not currently use this
        convergence tolerance. (default: 1e-3)
    method : str, optional
        BLaDE minimizer: ``'lbfg'``, ``'sd'``, ``'sdfd'``, or
        ``'sdmd'``. SDMD rejects invalid or uphill trial steps and
        restores the last accepted coordinates. (default: ``'sd'``)
    warn_restraints : bool, optional
        If True (default), warn if incompatible restraints are active.
    handle_interrupt : bool, optional
        If True (default), install a signal handler so Ctrl+C stops
        minimization gracefully instead of terminating Python.
    **kwargs : dict
        Additional minimization settings.

    Returns
    -------
    bool
        True if minimization completed normally, False if interrupted.

    Notes
    -----
    BLaDE does NOT support the following restraint types:
    - RESD (restrained distances)
    - IC (internal coordinate) restraints
    - DROPLET restraints

    BLaDE supports:
    - Harmonic restraints (full support)
    - Fix constraints (full support)
    - Dihedral restraints (full support)
    - NOE/PNOE (single atom selections)

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_blade(nstep=1000, tolgrd=1e-4)
    >>> minimize.run_blade(nstep=1000, method='sdmd')
    """
    import warnings
    import pycharmm.restraints as restraints
    import pycharmm.blade as blade

    method = method.lower()
    methods = ('lbfg', 'sd', 'sdfd', 'sdmd')
    if method not in methods:
        raise ValueError(
            f"Unknown BLaDE minimizer {method!r}; choose from {', '.join(methods)}"
        )

    # Check for incompatible restraints
    if warn_restraints:
        conflicts = restraints._state._check_backend_conflicts('blade')
        if conflicts:
            warnings.warn(
                f"Active restraints incompatible with BLaDE: {', '.join(conflicts)}. "
                f"These restraints will not be evaluated correctly in BLaDE minimization.",
                UserWarning, stacklevel=2
            )
        # Update backend state
        restraints.set_backend('blade')

    # Install signal handler for graceful Ctrl+C handling
    if handle_interrupt:
        blade.install_signal_handler()

    try:
        min_script = pycharmm.script.CommandScript(f'mini blade {method}',
                                                   nstep=nstep,
                                                   tolgrd=tolgrd,
                                                   **kwargs)
        min_script.run()

        # Check if minimization was interrupted
        interrupted = blade.check_interrupt()

    finally:
        # Always restore signal handler
        if handle_interrupt:
            blade.restore_signal_handler()
            blade.set_interrupt(0)  # Clear flag for next run

    return not interrupted


def run_conj(nstep: int = 100, ncgcyc: int = 100, pcut: float = 0.9999,
             prtmin: int = 1, lattice: bool = False, nocoords: bool = False,
             **kwargs):
    """Run Conjugate Gradient minimization

    Better convergence characteristics than steepest descent. Converges
    to minimum in N steps for quadratic energy surface (N = degrees of freedom).

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. (default: 100)
    ncgcyc : int
        Number of conjugate gradient cycles before algorithm restarts.
        (default: 100)
    pcut : float
        If cosine of angle between old and new P vector > PCUT, algorithm
        restarts. Prevents plodding down the same path. (default: 0.9999)
    prtmin : int
        Print level. <2: energy once per cycle. 2: energy for each
        evaluation plus method variables. (default: 1)
    lattice : bool
        With CRYSTAL, also optimize unit cell box size and/or shape.
        (default: False)
    nocoords : bool
        With CRYSTAL, only optimize unit cell (no coordinate changes).
        (default: False)
    **kwargs : dict
        Additional minimization settings including:
        - inbfrq: int - Non-bonded list update frequency (default: 50)
        - ihbfrq: int - Hydrogen bond list update frequency (default: 50)
        - nprint: int - Energy print frequency (default: 1)
        - tolenr: float - Energy change tolerance for exit
        - tolgrd: float - Average gradient tolerance for exit
        - tolitr: int - Max energy evaluations per step (default: 100)
        - step: float - Initial step size (default: 0.02)

    Notes
    -----
    More likely to generate numerical overflows than SD with poor conformations.
    Uses improved interpolation scheme and automatic step size selection.

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_conj(nstep=500, ncgcyc=100, tolgrd=1e-4)
    """
    cmd_kwargs = {'nstep': nstep, 'ncgcyc': ncgcyc, 'pcut': pcut,
                  'prtmin': prtmin}
    if lattice:
        cmd_kwargs['lattice'] = True
    if nocoords:
        cmd_kwargs['nocoords'] = True

    min_script = pycharmm.script.CommandScript('mini conj',
                                               **cmd_kwargs,
                                               **kwargs)
    min_script.run()


def run_powell(nstep: int = 100, lattice: bool = False,
               nocoords: bool = False, **kwargs):
    """Run Powell Conjugate Gradient minimization

    Improved efficiency over Fletcher-Reeves conjugate gradient method.
    Recommended when ABNR is not possible.

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. (default: 100)
    lattice : bool
        With CRYSTAL, also optimize unit cell box size and/or shape.
        (default: False)
    nocoords : bool
        With CRYSTAL, only optimize unit cell (no coordinate changes).
        (default: False)
    **kwargs : dict
        Additional minimization settings including:
        - inbfrq: int - Non-bonded list update frequency (default: 50)
        - ihbfrq: int - Hydrogen bond list update frequency (default: 50)
        - nprint: int - Energy print frequency (default: 1)
        - tolenr: float - Energy change tolerance for exit
        - tolgrd: float - Average gradient tolerance for exit
        - step: float - Initial step size (default: 0.02)

    Notes
    -----
    POWELL has no INBFRQ or IHBFRQ feature. Use CHARMM loops to mimic
    periodic updates. Harmonic constraint minimization with periodic
    updates is recommended for bad contacts or unlikely conformations.

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_powell(nstep=500, tolgrd=1e-4)
    """
    cmd_kwargs = {'nstep': nstep}
    if lattice:
        cmd_kwargs['lattice'] = True
    if nocoords:
        cmd_kwargs['nocoords'] = True

    min_script = pycharmm.script.CommandScript('mini powell',
                                               **cmd_kwargs,
                                               **kwargs)
    min_script.run()


def run_nrap(nstep: int = 100, tfreq: float = 1.0, sadd: int = 0, **kwargs):
    """Run Newton-Raphson minimization

    Solves Newton-Raphson equations iteratively by diagonalizing the
    second derivative matrix. Fast convergence for nearly quadratic potentials.

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. (default: 100)
    tfreq : float
        Smallest eigenvalue considered non-negative. Cubic fitting is
        applied to all eigenvalues smaller than this. (default: 1.0)
    sadd : int
        Order of saddle point to find. SADD=1 searches in opposite direction
        of most negative eigenvector (uphill) for transition states.
        (default: 0, normal minimization)
    **kwargs : dict
        Additional minimization settings including:
        - inbfrq: int - Non-bonded list update frequency (default: 50)
        - ihbfrq: int - Hydrogen bond list update frequency (default: 50)
        - nprint: int - Energy print frequency (default: 1)
        - tolenr: float - Energy change tolerance for exit
        - tolgrd: float - Average gradient tolerance for exit
        - step: float - Initial step size (default: 0.02)

    Notes
    -----
    Requires O(n²) storage and O(n³) computation time. Restricted to
    systems of about 200 atoms or less. IMAGES and SHAKE are unavailable.

    For transition state searches, slightly perturb the starting structure
    in the direction of the expected transition state.

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_nrap(nstep=100, tfreq=1.0)
    >>> minimize.run_nrap(nstep=50, sadd=1)  # Find transition state
    """
    cmd_kwargs = {'nstep': nstep, 'tfreq': tfreq}
    if sadd != 0:
        cmd_kwargs['sadd'] = sadd

    min_script = pycharmm.script.CommandScript('mini nrap',
                                               **cmd_kwargs,
                                               **kwargs)
    min_script.run()


def run_tn(nstep: int = 100, ncgcyc: int = 100, tolgrd: float = 0.0,
           prec: bool = False, anal: bool = True, rest: bool = True,
           sche: bool = False, sear: bool = False, iord: bool = False,
           perm: bool = False, **kwargs):
    """Run Truncated Newton (TNPACK) minimization

    Preconditioned linear conjugate-gradient technique for solving
    Newton equations. Exploits Hessian sparsity for preconditioning.

    Parameters
    ----------
    nstep : int
        Number of cycles of minimization. (default: 100)
    ncgcyc : int
        Number of conjugate gradient cycles. (default: 100)
    tolgrd : float
        Convergence tolerance. 0.0 sets to 1e-8, 1.0 sets to 1e-12.
        Controls multiple convergence tests (T1-T4). (default: 0.0)
    prec : bool
        Enable preconditioning. (default: False, NOPR)
    anal : bool
        Use analytic Hessian-vector products. False uses finite difference.
        (default: True, ANAL)
    rest : bool
        Use residual PCG truncation test. False uses quadratic (QUAT).
        (default: True, REST)
    sche : bool
        Enable scheduling subroutine. Turns on preconditioning when gradient
        is small, uses SD beforehand. (default: False, NOSC)
    sear : bool
        Enable optimal search-vector subroutine. Considers multiple descent
        directions, requires additional evaluations. (default: False, SROF)
    iord : bool
        Enable M reordering to minimize fill-in. Useful for large sparse M.
        (default: False, NOOR)
    perm : bool
        Indicates permutation array for reordering is known.
        (default: False, NOPM)
    **kwargs : dict
        Additional minimization settings including:
        - inbfrq: int - Non-bonded list update frequency (default: 50)
        - ihbfrq: int - Hydrogen bond list update frequency (default: 50)
        - nprint: int - Energy print frequency (default: 1)
        - tolenr: float - Energy change tolerance for exit

    Notes
    -----
    Compares favorably to ABNR in CPU time with numeric option.
    With analytic option, converges faster for small/medium systems
    (<400 atoms) and well-relaxed large systems. SHAKE unavailable.

    Examples
    --------
    >>> import pycharmm.minimize as minimize
    >>> minimize.run_tn(nstep=500, tolgrd=1e-6)
    >>> minimize.run_tn(nstep=500, prec=True, anal=True)  # With preconditioning
    """
    cmd_kwargs = {'nstep': nstep, 'ncgcyc': ncgcyc, 'tolgrd': tolgrd}
    # Boolean flags with keyword pairs
    if prec:
        cmd_kwargs['prec'] = True
    if not anal:
        cmd_kwargs['fdif'] = True  # FDIF is opposite of ANAL
    if not rest:
        cmd_kwargs['quat'] = True  # QUAT is opposite of REST
    if sche:
        cmd_kwargs['sche'] = True
    if sear:
        cmd_kwargs['sear'] = True
    if iord:
        cmd_kwargs['iord'] = True
    if perm:
        cmd_kwargs['perm'] = True

    min_script = pycharmm.script.CommandScript('mini tn',
                                               **cmd_kwargs,
                                               **kwargs)
    min_script.run()
