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

"""Finds and loads the CHARMM shared library

On importing pycharmm, this module looks for an environment variable named
`CHARMM_HOME` to find path to CHARMM shared library. 
The extension for the library is set
depending on the output of platform.system
"""


import ctypes
import os
import os.path
import platform


# CHARMM shared-library basenames, in preference order. Both the ``lib``-prefixed
# and bare forms are accepted so either build layout resolves.
_CHARMM_LIB_BASENAMES = ('libcharmm', 'charmm')


def charmm_lib_suffix(sys_name=None):
    """Platform shared-library extension: ``.dylib`` (macOS), ``.so`` (Linux),
    ``.dll`` (Windows). Defaults to ``.so`` for unknown platforms."""
    sys_name = sys_name or platform.system()
    return {'Darwin': '.dylib', 'Linux': '.so', 'Windows': '.dll'}.get(sys_name, '.so')


def _find_charmm_lib_in_dir(directory, suffix):
    """First existing ``{lib,}charmm<suffix>`` under *directory* or ``directory/lib``."""
    if not directory:
        return None
    for base_dir in (directory, os.path.join(directory, 'lib')):
        for base in _CHARMM_LIB_BASENAMES:
            candidate = os.path.join(base_dir, base + suffix)
            if os.path.isfile(candidate):
                return candidate
    return None


def _discover_repo_charmm_lib(suffix):
    """Walk up from this file for a ``setup/charmm`` dir holding ``libcharmm<suffix>``.

    Self-contained on purpose: importing ``mmml`` here would pull in jax/physnet
    at CHARMM-load time. Mirrors ``mmml...charmm_paths.default_repo_charmm_home``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        found = _find_charmm_lib_in_dir(os.path.join(here, 'setup', 'charmm'), suffix)
        if found:
            return found
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def resolve_charmm_lib_path(charmm_lib_dir=''):
    """Resolve the CHARMM shared library, cross-platform.

    Order: explicit ``charmm_lib_dir`` (``CHARMM_LIB_DIR``) → repo ``setup/charmm``
    auto-discovery → bare ``libcharmm<suffix>`` for the dynamic loader search path.
    """
    suffix = charmm_lib_suffix()
    if charmm_lib_dir:
        found = _find_charmm_lib_in_dir(charmm_lib_dir, suffix)
        # Fall back to the joined path even if missing, so CDLL raises against the
        # location the caller explicitly requested.
        return found or os.path.join(charmm_lib_dir, 'libcharmm' + suffix)
    return _discover_repo_charmm_lib(suffix) or ('libcharmm' + suffix)


class CharmmLib:
    def __init__(self, charmm_lib_dir=''):
        self.charmm_lib_name = resolve_charmm_lib_path(charmm_lib_dir)

        self.lib = None
        self.init_charmm()

        self.dlclose = ctypes.CDLL(None).dlclose  # does not work
        self.dlclose.argtypes = [ctypes.c_void_p]

    def __del__(self):
        self.del_charmm()


    def init_charmm(self):
        try:
            self.lib = ctypes.CDLL(self.charmm_lib_name)
        except OSError as exc:
            suffix = charmm_lib_suffix()
            raise OSError(
                f"Could not load the CHARMM shared library ({self.charmm_lib_name!r}).\n"
                f"Platform {platform.system()!r} expects a {suffix!r} library.\n"
                "Set CHARMM_LIB_DIR to the directory containing "
                f"libcharmm{suffix}, or place it under <repo>/setup/charmm.\n"
                f"Original error: {exc}"
            ) from exc
        self.lib.init_charmm()

    def del_charmm(self):
        if self.lib is None:
            return
        self.lib.del_charmm()  # initiates 'normal stop'
        # does not work
        # self.lib.dlclose(self.lib)


charmm_lib = CharmmLib(os.environ.get('CHARMM_LIB_DIR', ''))
charmm = charmm_lib.lib
