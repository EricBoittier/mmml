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

"""BLaDE (Basic LAmbda Dynamics Engine) interface for pycharmm.

This module provides functions to enable, configure, and control BLaDE
GPU-accelerated dynamics and energy calculations.

Functions
=========
- `enable` -- Enable BLaDE for GPU-accelerated calculations
- `disable` -- Disable BLaDE and return to CPU calculations
- `is_enabled` -- Check if BLaDE is currently enabled
- `energy` -- Run a single BLaDE energy calculation
- `check_restraints` -- Check active restraints for BLaDE compatibility
- `set_interrupt` -- Set interrupt flag to stop dynamics gracefully
- `check_interrupt` -- Check if interrupt flag is set

CHARMM Command Reference
========================
- `energy blade` → `blade.energy()`
- `energy blade off` → `blade.disable()`

Restraint Compatibility
=======================
BLaDE supports:
- Harmonic restraints (CONS HARM) - Full support
- Fix constraints (CONS FIX) - Full support
- Dihedral restraints (CONS DIHE) - Full support
- NOE/PNOE - Single atom selections supported

BLaDE does NOT support:
- RESD restraints
- IC restraints
- Droplet restraints

Examples
========
>>> import pycharmm.blade as blade
>>> import pycharmm.restraints as restraints

# Enable BLaDE with automatic restraint check
>>> blade.enable()

# Check if BLaDE is enabled
>>> blade.is_enabled()
True

# Run BLaDE energy calculation
>>> blade.energy()

# Disable BLaDE
>>> blade.disable()

# Check restraints before enabling (without actually enabling)
>>> issues = blade.check_restraints()
>>> if issues:
...     print("Warning:", issues)
"""

import warnings
import ctypes
import signal
from pycharmm.lingo import charmm_script
import pycharmm.restraints as restraints
from pycharmm.loader import lib


# Module-level state
_blade_enabled = False
_signal_handler_installed = False
_original_sigint_handler = None
_sigint_count = 0
_sigint_last_time = 0.0
_RAPID_CTRL_C_THRESHOLD = 3  # Number of Ctrl+C to force exit
_RAPID_CTRL_C_WINDOW = 2.0  # Seconds window for rapid presses


class BladeRestraintWarning(UserWarning):
    """Warning issued when active restraints may be incompatible with BLaDE."""
    pass


class BladeEngineError(Exception):
    """Error related to BLaDE engine operations."""
    pass


class BladeInterrupted(Exception):
    """Exception raised when BLaDE dynamics is interrupted by Ctrl+C."""
    pass


def _setup_interrupt_functions():
    """Set up ctypes bindings for BLaDE interrupt functions."""
    chm = lib

    # Set up function signatures
    chm.blade_set_interrupt.argtypes = [ctypes.c_int]
    chm.blade_set_interrupt.restype = None

    chm.blade_check_interrupt.argtypes = []
    chm.blade_check_interrupt.restype = ctypes.c_int

    chm.blade_install_signal_handler.argtypes = []
    chm.blade_install_signal_handler.restype = None

    chm.blade_restore_signal_handler.argtypes = []
    chm.blade_restore_signal_handler.restype = None


def _python_sigint_handler(signum, frame):
    """Python signal handler that sets the BLaDE interrupt flag.

    Tracks rapid Ctrl+C presses: if pressed multiple times within a short
    window, force-exits Python immediately instead of waiting for graceful stop.
    """
    import sys
    import time

    global _sigint_count, _sigint_last_time, _original_sigint_handler

    current_time = time.time()

    # Check if this is a rapid press (within the time window)
    if current_time - _sigint_last_time < _RAPID_CTRL_C_WINDOW:
        _sigint_count += 1
    else:
        _sigint_count = 1

    _sigint_last_time = current_time

    # If pressed rapidly multiple times, force exit
    if _sigint_count >= _RAPID_CTRL_C_THRESHOLD:
        print(f"\n*** {_RAPID_CTRL_C_THRESHOLD}x Ctrl+C detected - FORCE EXITING ***",
              flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original handler and re-raise to let Python handle it
        if _original_sigint_handler is not None:
            signal.signal(signal.SIGINT, _original_sigint_handler)
        # Re-raise the signal to trigger default behavior (KeyboardInterrupt)
        raise KeyboardInterrupt("Force exit by rapid Ctrl+C")

    remaining = _RAPID_CTRL_C_THRESHOLD - _sigint_count
    print(f"\n*** Ctrl+C pressed - requesting BLaDE to stop gracefully "
          f"(press {remaining}x more rapidly to force quit) ***", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    set_interrupt(1)
    # Also trigger C++ handler
    lib.blade_install_signal_handler()


def set_interrupt(value=1):
    """Set the BLaDE interrupt flag.

    This can be used to gracefully stop dynamics from another thread
    or signal handler.

    Parameters
    ----------
    value : int, optional
        1 to request interrupt, 0 to clear. Default 1.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.set_interrupt()  # Request dynamics to stop
    >>> blade.set_interrupt(0)  # Clear interrupt flag
    """
    _setup_interrupt_functions()
    lib.blade_set_interrupt(ctypes.c_int(value))


def check_interrupt():
    """Check if the BLaDE interrupt flag is set.

    Returns
    -------
    bool
        True if interrupt was requested, False otherwise.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> if blade.check_interrupt():
    ...     print("Dynamics was interrupted")
    """
    _setup_interrupt_functions()
    return bool(lib.blade_check_interrupt())


def install_signal_handler():
    """Install BLaDE's SIGINT (Ctrl+C) handler.

    When installed, pressing Ctrl+C during BLaDE dynamics will set an
    interrupt flag that causes dynamics to stop gracefully at the next
    step, rather than immediately terminating the Python process.

    Pressing Ctrl+C multiple times rapidly (3x within 2 seconds) will
    force-exit Python immediately with KeyboardInterrupt.

    This is automatically called by dynamics() but can be called manually
    for custom workflows.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.install_signal_handler()
    >>> # ... run dynamics ...
    >>> blade.restore_signal_handler()
    """
    global _signal_handler_installed, _original_sigint_handler
    global _sigint_count, _sigint_last_time
    _setup_interrupt_functions()

    # Reset rapid Ctrl+C counter
    _sigint_count = 0
    _sigint_last_time = 0.0

    # Install both Python-level and C++-level signal handlers
    # Python handler sets the flag immediately when Ctrl+C is pressed
    _original_sigint_handler = signal.signal(signal.SIGINT, _python_sigint_handler)

    # Also install C++ level handler as backup
    lib.blade_install_signal_handler()

    # Clear any previous interrupt flag
    set_interrupt(0)
    _signal_handler_installed = True


def restore_signal_handler():
    """Restore the original SIGINT handler.

    This restores Python's default Ctrl+C behavior. Called automatically
    after dynamics() completes.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.restore_signal_handler()
    """
    global _signal_handler_installed, _original_sigint_handler
    global _sigint_count, _sigint_last_time
    _setup_interrupt_functions()

    # Reset rapid Ctrl+C counter
    _sigint_count = 0
    _sigint_last_time = 0.0

    # Restore original Python signal handler
    if _original_sigint_handler is not None:
        signal.signal(signal.SIGINT, _original_sigint_handler)
        _original_sigint_handler = None

    # Also restore C++ level handler
    lib.blade_restore_signal_handler()
    _signal_handler_installed = False


def check_restraints():
    """Check if active restraints are compatible with BLaDE.

    Returns a list of restraint types that are incompatible with BLaDE,
    or an empty list if all active restraints are supported.

    Returns
    -------
    list[str]
        List of incompatible restraint type names, empty if all compatible.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> issues = blade.check_restraints()
    >>> if issues:
    ...     print(f"Incompatible restraints: {issues}")
    """
    from pycharmm.restraints import _state

    conflicts = _state._check_backend_conflicts('blade')
    return conflicts


def enable(verb=None, warn_restraints=True, raise_on_incompatible=False):
    """Enable BLaDE for GPU-accelerated energy and dynamics calculations.

    Before enabling BLaDE, this function checks for active restraints that
    may be incompatible. By default, a warning is issued for incompatible
    restraints, but BLaDE is still enabled. Set `raise_on_incompatible=True`
    to raise an error instead.

    Parameters
    ----------
    verb : int, optional
        Verbosity level for BLaDE debug output:
        - 0: Suppress debug output (default)
        - 1: Basic debug output
        - 2: Detailed debug output
        None means don't specify (use CHARMM default).
    warn_restraints : bool, optional
        If True (default), issue a warning if incompatible restraints are active.
    raise_on_incompatible : bool, optional
        If True, raise BladeEngineError instead of warning. Default False.

    Raises
    ------
    BladeEngineError
        If raise_on_incompatible=True and incompatible restraints are active.

    Notes
    -----
    Equivalent CHARMM commands:
    - `BLADE ON` - Enable BLaDE
    - `BLADE VERB level` - Set verbosity level

    The following restraint types are NOT supported in BLaDE:
    - RESD (restrained distances)
    - IC (internal coordinate restraints)
    - DROPLET restraints

    Examples
    --------
    >>> import pycharmm.blade as blade

    # Enable with default warning behavior
    >>> blade.enable()

    # Enable with verbose debug output
    >>> blade.enable(verb=1)

    # Enable and raise error if restraints are incompatible
    >>> blade.enable(raise_on_incompatible=True)

    # Enable silently (no warnings)
    >>> blade.enable(warn_restraints=False)
    """
    global _blade_enabled

    # Check for incompatible restraints
    conflicts = check_restraints()
    if conflicts:
        msg = (
            f"Active restraints incompatible with BLaDE: {', '.join(conflicts)}. "
            f"These restraints will not be evaluated correctly on GPU. "
            f"Consider using CPU energy calculations or removing these restraints."
        )
        if raise_on_incompatible:
            raise BladeEngineError(msg)
        elif warn_restraints:
            warnings.warn(msg, BladeRestraintWarning, stacklevel=2)

    # Update restraints state to track backend
    try:
        restraints.set_backend('blade')
    except restraints.IncompatibleBackendError:
        # If restraints module raises, we still enable BLaDE but warn
        if not raise_on_incompatible:
            pass  # Warning already issued above
        else:
            raise

    # Enable BLaDE via CHARMM (BLADE ON overrides domdec and openmm)
    charmm_script("BLADE ON")
    _blade_enabled = True

    # Set verbosity if specified
    if verb is not None:
        set_verbosity(verb)


def set_verbosity(level):
    """Set BLaDE verbosity level for debug output.

    Parameters
    ----------
    level : int
        Verbosity level:
        - 0: Suppress debug output (default)
        - 1: Detailed debug output

    Notes
    -----
    Equivalent CHARMM command: `BLADE VERB level`

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.enable()
    >>> blade.set_verbosity(1)  # Enable detailed debug output
    """
    if level not in (0, 1):
        raise ValueError(f"Verbosity level must be 0 or 1, got {level}")
    charmm_script(f"BLADE VERB {level}")


def disable():
    """Disable BLaDE and return to CPU-based calculations.

    Notes
    -----
    Equivalent CHARMM command: `BLADE OFF`

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.disable()
    """
    global _blade_enabled

    charmm_script("BLADE OFF")
    _blade_enabled = False

    # Reset backend to standard
    restraints.set_backend('standard')


def run_script(filename):
    """Read and execute a BLaDE script file.

    Parameters
    ----------
    filename : str
        Path to BLaDE script file.

    Notes
    -----
    Equivalent CHARMM command: `BLADE FILE filename`

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.run_script("my_blade_options.inp")
    """
    charmm_script(f"BLADE FILE {filename}")


def is_enabled():
    """Check if BLaDE is currently enabled.

    Returns
    -------
    bool
        True if BLaDE is enabled, False otherwise.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.enable()
    >>> blade.is_enabled()
    True
    >>> blade.disable()
    >>> blade.is_enabled()
    False
    """
    return _blade_enabled


def energy(show=True, abic=False):
    """Calculate energy using BLaDE.

    Parameters
    ----------
    show : bool, optional
        If True (default), display energy breakdown.
    abic : bool, optional
        If True, use "Assume BLaDE Is Current" mode. This skips
        BLaDE reinitialization and assumes coordinates are already
        current on the GPU. Useful for repeated energy calls where
        coordinates haven't changed via standard CHARMM. Default False.

    Notes
    -----
    Equivalent CHARMM commands:
    - `energy blade` - Standard BLaDE energy
    - `energy blade abic` - BLaDE energy assuming GPU coords current

    If BLaDE is not already enabled, this function will enable it.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> blade.energy()

    # Skip reinitialization for repeated calls
    >>> blade.energy(abic=True)
    """
    global _blade_enabled

    if not _blade_enabled:
        enable()

    cmd = "energy blade"
    if abic:
        cmd += " abic"
    charmm_script(cmd)

    if show:
        import pycharmm.energy as eng
        eng.show()


def dynamics(nstep=1000, timestep=0.002, finalt=300.0,
             prmc=False, iprs=50, pref=1.0, prdv=100.0,
             abic=False, warn_restraints=True, handle_interrupt=True,
             raise_on_interrupt=False, **kwargs):
    """Run BLaDE dynamics with optional constant pressure.

    Parameters
    ----------
    nstep : int, optional
        Number of dynamics steps. Default 1000.
    timestep : float, optional
        Time step in picoseconds. Default 0.002.
    finalt : float, optional
        Temperature for Langevin thermostat in Kelvin. Default 300.0.
    prmc : bool, optional
        Enable pressure Monte Carlo barostat. Default False (NVT).
    iprs : int, optional
        Frequency for volume change attempts (steps). Default 50.
    pref : float, optional
        Reference pressure in atmospheres. Default 1.0.
    prdv : float, optional
        Standard deviation for volume changes in Å³. Default 100.0.
    abic : bool, optional
        If True, use "Assume BLaDE Is Current" mode. This skips
        BLaDE reinitialization and assumes coordinates are already
        current on the GPU. Default False.
    warn_restraints : bool, optional
        If True (default), warn if incompatible restraints are active.
    handle_interrupt : bool, optional
        If True (default), install a signal handler so Ctrl+C stops
        dynamics gracefully instead of terminating Python immediately.
    raise_on_interrupt : bool, optional
        If True and dynamics was interrupted by Ctrl+C, raise
        BladeInterrupted exception. Default False (just return).
    **kwargs : dict
        Additional dynamics options passed to CHARMM.

    Returns
    -------
    bool
        True if dynamics completed normally, False if interrupted.

    Raises
    ------
    BladeInterrupted
        If raise_on_interrupt=True and dynamics was interrupted.

    Notes
    -----
    Equivalent CHARMM command:
    `DYNAmics [opts] BLADE [PRMC IPRS iprs PREF pref PRDV prdv] [ABIC] FINALT finalt`

    BLaDE supports only Langevin thermostat (or constant energy) and
    Monte Carlo barostat (or constant volume). Drag coefficients are
    controlled via `scalar fbeta` for spatial degrees and `BLOCK LDIN`
    for alchemical degrees.

    The handle_interrupt=True option allows you to press Ctrl+C during
    dynamics and have BLaDE stop gracefully at the next step, preserving
    the current state. Without this, Ctrl+C would terminate Python
    immediately, potentially corrupting output files.

    Examples
    --------
    >>> import pycharmm.blade as blade

    # NVT dynamics at 300K
    >>> blade.dynamics(nstep=10000, finalt=300.0)

    # NPT dynamics with pressure barostat
    >>> blade.dynamics(nstep=10000, finalt=300.0, prmc=True, pref=1.0)

    # Dynamics with interrupt handling that raises exception
    >>> try:
    ...     blade.dynamics(nstep=100000, raise_on_interrupt=True)
    ... except blade.BladeInterrupted:
    ...     print("User interrupted dynamics")

    # Dynamics assuming BLaDE already has current coords
    >>> blade.dynamics(nstep=1000, abic=True)
    """
    global _blade_enabled

    # Check restraint compatibility
    if warn_restraints:
        conflicts = check_restraints()
        if conflicts:
            warnings.warn(
                f"Active restraints incompatible with BLaDE: {', '.join(conflicts)}. "
                f"These restraints will not be evaluated correctly.",
                BladeRestraintWarning, stacklevel=2
            )

    if not _blade_enabled:
        enable(warn_restraints=False)  # Already warned above

    # Install signal handler for graceful Ctrl+C handling
    if handle_interrupt:
        install_signal_handler()

    try:
        # Build dynamics command
        cmd_parts = [f"dynamics nstep {nstep} timestep {timestep}"]

        # Add any additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(key)
            else:
                cmd_parts.append(f"{key} {value}")

        # Add BLADE keyword and options
        cmd_parts.append("blade")

        if prmc:
            cmd_parts.append(f"prmc iprs {iprs} pref {pref} prdv {prdv}")

        if abic:
            cmd_parts.append("abic")

        cmd_parts.append(f"finalt {finalt}")

        cmd = " ".join(cmd_parts)
        charmm_script(cmd)

        # Check if dynamics was interrupted
        interrupted = check_interrupt()

    finally:
        # Always restore signal handler
        if handle_interrupt:
            restore_signal_handler()
            set_interrupt(0)  # Clear flag for next run

    if interrupted:
        if raise_on_interrupt:
            raise BladeInterrupted("BLaDE dynamics interrupted by user (Ctrl+C)")
        return False

    return True


def get_supported_restraints():
    """Get information about restraint support in BLaDE.

    Returns
    -------
    dict
        Dictionary with 'supported' and 'unsupported' lists of restraint types,
        plus 'notes' with additional compatibility information.

    Examples
    --------
    >>> import pycharmm.blade as blade
    >>> info = blade.get_supported_restraints()
    >>> print("Supported:", info['supported'])
    >>> print("Unsupported:", info['unsupported'])
    """
    return {
        'supported': [
            'HARMONIC (CONS HARM) - Full support for all harmonic restraints',
            'FIX (CONS FIX) - Full support for fixed atom constraints',
            'DIHE (CONS DIHE) - Full support for dihedral restraints',
            'NOE/PNOE - Single atom selections supported',
            'SCAT - Constrained atom scaling (via BLOCK)'
        ],
        'unsupported': [
            'RESD - Restrained distances not implemented',
            'IC - Internal coordinate restraints not implemented',
            'DROPLET - Droplet restraints not implemented'
        ],
        'notes': {
            'NOE': 'Single atom/atoms NOE and PNOE supported. '
                   'Complex multi-atom averaging may have limitations.',
        }
    }
