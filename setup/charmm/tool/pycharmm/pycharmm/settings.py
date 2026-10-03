# pyCHARMM: molecular dynamics in python with CHARMM
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

"""Functions to change CHARMM print level (`PRNLev`), warning level (`WRNLev`)
and bomb level (`BOMBlev`)

See CHARMM documentation [miscom](<https://academiccharmm.org/documentation/version/c47b1/miscom>)
for more information

Functions
=========

- `set_verbosity` -- set the CHARMM library's verbosity level
- `set_warn_level` -- set the CHARMM library's warning level
- `set_bomb_level` -- set the CHARMM library's bomb level
- `print_config` -- print out build information

Examples
========
>>> import pycharmm
>>> import pycharmm.settings as settings

The following command is equivalent to CHARMM command `PRNLev 0`
>>> settings.set_verbosity(0)
"""

import ctypes
from contextlib import contextmanager
from pycharmm.loader import lib


# set charmm's verbosity
def set_verbosity(level):
    """change verbosity of CHARMM library

    Parameters
    ----------
    level : int
            the new verbosity level desired

    Returns
    -------
    old_level : int
                old verbosity level
    """
    level = ctypes.c_int(level)
    old_level = lib.stream_set_prnlev(ctypes.byref(level))
    return old_level


# set charmm's warning level
def set_warn_level(level):
    """change CHARMM's warning level

    Parameters
    ----------
    level : int
            the new warning level desired

    Returns
    -------
    old_level : int
                old warning level
    """
    level = ctypes.c_int(level)
    old_level = lib.stream_set_wrnlev(ctypes.byref(level))
    return old_level


# set charmm's bomb level
def set_bomb_level(level):
    """change CHARMM's bomb level

    Parameters
    ----------
    level : int
            the new bomb level desired

    Returns
    -------
    old_level : int
                old bomb level
    """
    level = ctypes.c_int(level)
    old_level = lib.stream_set_bomlev(ctypes.byref(level))
    return old_level


# echo of direct pyCHARMM api calls into the CHARMM output
_api_echo = False


def set_api_echo(enabled=True):
    """turn echoing of direct pyCHARMM api calls on or off

    Some pyCHARMM functions call a Fortran api entry point directly,
    bypassing the CHARMM script interpreter, so nothing in the CHARMM
    output shows that the command ran. When echo is on, each such call
    writes a line like ``PYCHARMM>  rtf.get_num_residues`` to the CHARMM
    output, which helps confirm from the output that a command was
    issued when debugging a pyCHARMM script. Off by default.

    Parameters
    ----------
    enabled : bool
              True to echo traced api calls into the CHARMM output,
              False to silence them

    Returns
    -------
    previous : bool
               the echo setting in effect before this call
    """
    global _api_echo
    previous = _api_echo
    _api_echo = bool(enabled)
    return previous


def get_api_echo():
    """report whether echoing of direct pyCHARMM api calls is on

    Returns
    -------
    enabled : bool
              True if traced api calls are being echoed to the output
    """
    return _api_echo


# print build information
def print_config():
    """print build information

    This routine prints the command line arguments to both
    the configure script and cmake and
    prints the original source code directory
    """
    lib.print_config()


# Context managers for temporary level changes

@contextmanager
def bomb_level(level):
    """Context manager to temporarily set CHARMM's bomb level.

    The bomb level controls when CHARMM stops execution on errors.
    Lower values are more permissive (suppress more errors).

    Parameters
    ----------
    level : int
        The bomb level to use within the context.
        Common values:
        - -5: Suppress most errors (very permissive)
        - -1: Suppress minor errors
        - 0: Default CHARMM behavior
        - 5: Very strict

    Yields
    ------
    old_level : int
        The previous bomb level.

    Examples
    --------
    >>> import pycharmm.settings as settings
    >>> import pycharmm.lingo as lingo

    # Temporarily suppress errors while reading an optional file
    >>> with settings.bomb_level(-1):
    ...     lingo.charmm_script('open read unit 10 name maybe_missing.pdb')
    ...     lingo.charmm_script('read coor pdb unit 10')

    # Bomb level is automatically restored after the with block
    """
    old_level = set_bomb_level(level)
    try:
        yield old_level
    finally:
        set_bomb_level(old_level)


@contextmanager
def warn_level(level):
    """Context manager to temporarily set CHARMM's warning level.

    The warning level controls which warnings are printed.
    Lower values suppress more warnings.

    Parameters
    ----------
    level : int
        The warning level to use within the context.

    Yields
    ------
    old_level : int
        The previous warning level.

    Examples
    --------
    >>> import pycharmm.settings as settings

    >>> with settings.warn_level(-5):
    ...     # Suppress warnings during noisy operations
    ...     pycharmm.read.prm('file.prm', flex=True)
    """
    old_level = set_warn_level(level)
    try:
        yield old_level
    finally:
        set_warn_level(old_level)


@contextmanager
def verbosity(level):
    """Context manager to temporarily set CHARMM's verbosity level.

    Controls how much output CHARMM produces.

    Parameters
    ----------
    level : int
        The verbosity level to use within the context.
        Lower values produce less output.

    Yields
    ------
    old_level : int
        The previous verbosity level.

    Examples
    --------
    >>> import pycharmm.settings as settings

    >>> with settings.verbosity(0):
    ...     # Silent operation
    ...     pycharmm.minimize.run_sd(nstep=100)
    """
    old_level = set_verbosity(level)
    try:
        yield old_level
    finally:
        set_verbosity(old_level)


@contextmanager
def error_levels(bomb=None, warn=None, print_level=None):
    """Context manager to temporarily set multiple CHARMM error levels.

    Convenience function to set bomb level, warning level, and verbosity
    together. Only specified levels are changed.

    Parameters
    ----------
    bomb : int, optional
        Bomb level (controls when CHARMM stops on errors).
    warn : int, optional
        Warning level (controls which warnings are shown).
    print_level : int, optional
        Print/verbosity level.

    Examples
    --------
    >>> import pycharmm.settings as settings

    >>> with settings.error_levels(bomb=-1, warn=-5, print_level=0):
    ...     # Quiet and permissive operation
    ...     pycharmm.read.prm('parameters.prm', flex=True)
    """
    old_bomb = set_bomb_level(bomb) if bomb is not None else None
    old_warn = set_warn_level(warn) if warn is not None else None
    old_print = set_verbosity(print_level) if print_level is not None else None
    try:
        yield
    finally:
        if old_bomb is not None:
            set_bomb_level(old_bomb)
        if old_warn is not None:
            set_warn_level(old_warn)
        if old_print is not None:
            set_verbosity(old_print)
