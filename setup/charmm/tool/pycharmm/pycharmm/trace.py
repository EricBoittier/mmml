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

"""Echo direct pyCHARMM api calls into the CHARMM output file

Many pyCHARMM operations build a CHARMM command and run it through the
interpreter (`pycharmm.lingo.charmm_script`), so they already appear in
the CHARMM output. Others call a Fortran C-API entry point directly and
leave no trace. Turning api echo on makes those direct calls announce
themselves in the CHARMM output file, which helps when debugging a
pyCHARMM script and you want to confirm from the output that a command
actually ran.

The on/off control lives in `pycharmm.settings`
(`set_api_echo` / `get_api_echo`), alongside the other CHARMM output
controls. This module provides the decorator that api wrappers use to
participate, and the low-level write. Echo is off by default; when on,
each traced call writes a line such as

    PYCHARMM>  rtf.get_residue_names

to the CHARMM output, via the api_trace Fortran routine.

Functions
=========
- `echo` -- write one api-call label to the CHARMM output (if echo is on)
- `traced` -- decorator that echoes a function's label when echo is on

Example
=======
>>> from pycharmm import settings, rtf
>>> settings.set_api_echo(True)
>>> n = rtf.get_num_residues()   # writes ' PYCHARMM>  rtf.get_num_residues'
"""

import ctypes
import functools

from pycharmm.loader import lib
import pycharmm.settings as settings


def echo(label):
    """write one api-call label to the CHARMM output file, if echo is on

    A no-op when api echo is off (see `pycharmm.settings.set_api_echo`).

    Parameters
    ----------
    label : str
            text to echo, normally the api function name
    """
    if not settings.get_api_echo():
        return
    data = str(label).encode('ascii', errors='replace')
    lib.api_trace(data, ctypes.c_int(len(data)))


def traced(label=None):
    """decorator that echoes a function's label to CHARMM output when echo is on

    Wrap a pyCHARMM api function so that, whenever api echo is enabled
    (see `pycharmm.settings.set_api_echo`), calling it first writes its
    label to the CHARMM output file. When echo is off the wrapper adds
    no output and negligible overhead.

    Parameters
    ----------
    label : str or None
            the label to echo; defaults to the wrapped function's name

    Returns
    -------
    decorator : callable
                a decorator that wraps the target function
    """
    def decorator(fn):
        name = label if label is not None else fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            echo(name)
            return fn(*args, **kwargs)

        return wrapper
    return decorator
