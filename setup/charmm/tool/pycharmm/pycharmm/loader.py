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
`CHARMM_LIB_DIR` to find path to CHARMM shared library.
The extension for the library is set
depending on the output of platform.system
"""


import ctypes
import os
import os.path
import platform
import sys
import warnings

from .dimens import dimens as dimens_settings


class CharmmLibraryLoadError(RuntimeError, OSError):
    """``libcharmm`` could not be loaded (missing file, bad path, missing symbol).

    MMML patch: c52a1 loads the library lazily and raised a plain RuntimeError
    here. The pre-c52a1 loader raised OSError at import, and MMML's optional-CHARMM
    paths (``except (ImportError, OSError)``) rely on that, so this error is both.
    """


class _CharmmLibLoader:
    def __init__(self, charmm_lib_dir=''):
        self._lib = None  # Ensure this is always set first
        # Optional base MPI communicator (Fortran handle from mpi4py's
        # comm.py2f()) applied at init via set_charmm_comm(); None = default.
        # _user_comm keeps a reference to the mpi4py Comm object alive so
        # mpi4py cannot MPI_Comm_free() it (on garbage collection) between
        # set_mpi_comm() and the deferred library init that consumes the
        # handle -- otherwise the stored handle would dangle.
        self._user_comm_handle = None
        self._user_comm = None
        # MMML installs ``libcharmm.so`` / ``.dylib``. Upstream c52a1's default
        # basename is ``libchmm``; accept either when a directory is searched.
        if os.environ.get('charmm_lib') is None:
            basenames = ("libcharmm", "libchmm")
        else:
            basenames = (os.path.expandvars("lib${charmm_lib}"),)
        sys_name = platform.system()
        if sys_name == 'Darwin':
            suffix = '.dylib'
        elif sys_name == 'Windows':
            suffix = '.dll'
        else:
            suffix = '.so'
        self.charmm_lib_name = basenames[0] + suffix

        search_dirs = []
        if charmm_lib_dir:
            search_dirs.append(charmm_lib_dir)
        for env_name in ("CHARMM_LIB_DIR", "CHARMM_HOME"):
            env_dir = os.environ.get(env_name) or ""
            if env_dir:
                search_dirs.append(env_dir)
        default_lib_dir = os.path.expandvars("${CMAKE_INSTALL_PREFIX}")
        if default_lib_dir:
            search_dirs.append(os.path.join(default_lib_dir, "lib"))
        for directory in search_dirs:
            found = None
            for base_dir in (directory, os.path.join(directory, "lib")):
                for base in basenames:
                    candidate = os.path.join(base_dir, base + suffix)
                    if os.path.isfile(candidate):
                        found = candidate
                        break
                if found:
                    break
            if found:
                self.charmm_lib_name = found
                break

        # Validate library path early and warn if issues detected
        self._validate_library_path(charmm_lib_dir)

        # MMML patch: dlopen the library now so ``import pycharmm`` raises
        # OSError when libcharmm is missing, as it did before c52a1. MMML's
        # optional-CHARMM paths (``try: import pycharmm.X`` / ``except
        # (ImportError, OSError)``) depend on that. Only the dlopen is eager:
        # ``init_charmm`` still runs on first use, so set_mpi_comm() works.
        self._dll = None
        try:
            self._dll = ctypes.CDLL(self.charmm_lib_name)
        except OSError as e:
            raise CharmmLibraryLoadError(
                f"Failed to load CHARMM shared library '{self.charmm_lib_name}'. "
                f"Ensure CHARMM_LIB_DIR environment variable is set correctly "
                f"and the library file exists: {e}"
            ) from e

        self.dlclose = ctypes.CDLL(None).dlclose  # does not work
        self.dlclose.argtypes = [ctypes.c_void_p]

    def _validate_library_path(self, charmm_lib_dir):
        """Check if the CHARMM library is accessible and warn early if not.

        This preserves lazy loading while giving users early feedback about
        potential configuration issues.
        """
        lib_path = self.charmm_lib_name

        # Check if the library file exists
        if not os.path.isabs(lib_path):
            # Library name is not an absolute path - it wasn't found at expected location
            env_lib_dir = os.environ.get('CHARMM_LIB_DIR', '')
            if not env_lib_dir:
                warnings.warn(
                    "CHARMM_LIB_DIR environment variable is not set. "
                    "CHARMM library may not be found when needed. "
                    "Set CHARMM_LIB_DIR to the directory containing the CHARMM shared library.",
                    UserWarning,
                    stacklevel=3
                )
            else:
                expected_path = os.path.join(env_lib_dir, lib_path)
                if not os.path.exists(expected_path):
                    warnings.warn(
                        f"CHARMM library not found at expected path: {expected_path}. "
                        f"Verify that CHARMM_LIB_DIR ('{env_lib_dir}') is correct "
                        f"and contains the CHARMM shared library.",
                        UserWarning,
                        stacklevel=3
                    )
        elif not os.path.exists(lib_path):
            # Absolute path but file doesn't exist
            warnings.warn(
                f"CHARMM library not found at: {lib_path}. "
                "Verify CHARMM_LIB_DIR is set correctly.",
                UserWarning,
                stacklevel=3
            )
        elif not os.access(lib_path, os.R_OK):
            # File exists but is not readable
            warnings.warn(
                f"CHARMM library at {lib_path} is not readable. "
                "Check file permissions.",
                UserWarning,
                stacklevel=3
            )

    def __del__(self):
        # Only call del_charmm if _lib attribute exists
        if hasattr(self, "_lib"):
            self.del_charmm()

    def _initialize_charmm_library(self):
        if self._lib is None:
            try:
                print("Initializing CHARMM library...")
                self._lib = (
                    self._dll if self._dll is not None
                    else ctypes.CDLL(self.charmm_lib_name)
                )
                dimens_settings.communicate_dimens(self._lib)
                # Hand CHARMM the caller-chosen base communicator (if any)
                # BEFORE init_charmm(), which is where it is adopted.
                if self._user_comm_handle is not None:
                    self._lib.set_charmm_comm.argtypes = [ctypes.c_int]
                    self._lib.set_charmm_comm.restype = None
                    self._lib.set_charmm_comm(int(self._user_comm_handle))
                self._lib.init_charmm()
            except OSError as e:
                self._lib = None
                raise CharmmLibraryLoadError(
                    f"Failed to load CHARMM shared library '{self.charmm_lib_name}'. "
                    f"Ensure CHARMM_LIB_DIR environment variable is set correctly "
                    f"and the library file exists: {e}"
                ) from e
            except Exception as e:
                self._lib = None
                raise RuntimeError(
                    f"Failed to initialize CHARMM library '{self.charmm_lib_name}': {e}"
                ) from e

    @property
    def handle(self):
        if self._lib is None:
            self._initialize_charmm_library()
        return self._lib

    def del_charmm(self):
        if hasattr(self, "_lib") and self._lib is not None:
            # Null _lib first so a second call (explicit close + interpreter
            # __del__) cannot re-run del_charmm -> stopch -> PARFIN a second
            # time on already-torn-down CHARMM state.
            lib = self._lib
            self._lib = None
            try:
                lib.del_charmm()
            except Exception as e:
                # Log but don't raise during cleanup - destructor should not throw
                print(f"[pycharmm.loader] WARNING: Failed to finalize CHARMM library: {e}",
                      file=sys.stderr)


_loader = _CharmmLibLoader(os.environ.get('CHARMM_LIB_DIR', ''))


def is_initialized():
    """Report whether the CHARMM shared library is up.

    Returns
    -------
    bool
        True once the library has been loaded and ``init_charmm`` has run.
    """
    return _loader._lib is not None


def initialize():
    """Load and initialize the CHARMM library now, instead of on first use.

    Initialization is normally lazy. Force it when the timing matters --
    chiefly under MPI, where adopting a host-supplied communicator is
    collective across that communicator, so every rank has to reach
    initialization together. Calling this explicitly keeps that
    collective at a known point rather than wherever the first CHARMM
    call happens to fall on each rank.

    Does nothing if CHARMM is already initialized.

    Returns
    -------
    None
    """
    # Call the initializer directly rather than touching the `handle`
    # property for its side effect: a bare attribute expression reads as
    # dead code (and lints as one).  _initialize_charmm_library is already
    # a no-op once the library is up.
    _loader._initialize_charmm_library()


def set_mpi_comm(comm):
    """Choose the base MPI communicator CHARMM runs on (embedded/pyCHARMM).

    Must be called BEFORE any CHARMM/pyCHARMM operation (i.e. before the
    library initializes).  ``comm`` is an mpi4py communicator; CHARMM adopts
    it as its base communicator, so ``N`` mpi4py groups of ``M`` ranks yield
    an ``N x (M-node CHARMM)`` layout (``?numnode == M`` within each group).

    Typical uses::

        # N independent serial CHARMMs (this is also the default if you
        # never call set_mpi_comm):
        set_mpi_comm(MPI.COMM_SELF)

        # N replicas, each an M-node parallel CHARMM:
        sub = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() // M,
                                   key=MPI.COMM_WORLD.Get_rank())
        set_mpi_comm(sub)

    Passing ``MPI.COMM_WORLD`` makes one parallel CHARMM over all ranks.
    Passing ``None`` clears a previous selection and reverts to the default.

    The communicator object is retained until the library initializes, so a
    communicator you build inline (``set_mpi_comm(COMM_WORLD.Split(...))``)
    stays valid even though CHARMM does not adopt it until its first call.
    """
    if _loader._lib is not None:
        raise RuntimeError(
            "set_mpi_comm() must be called before CHARMM is initialized "
            "(before the first pycharmm/CHARMM call).")
    if comm is None:
        _loader._user_comm = None
        _loader._user_comm_handle = None
        return
    py2f = getattr(comm, "py2f", None)
    if not callable(py2f):
        raise TypeError(
            "set_mpi_comm() expects an mpi4py communicator (with a py2f() "
            f"method) or None, got {type(comm).__name__}.")
    # Keep the Comm object alive (see _CharmmLibLoader.__init__) and store the
    # Fortran handle the C entry point (set_charmm_comm) consumes at init.
    _loader._user_comm = comm
    _loader._user_comm_handle = int(py2f())


class _LazyLib:
    """Proxy that defers CHARMM library initialization until first use.

    Importing this object does NOT initialize the Fortran runtime.
    Initialization is triggered only when an attribute (i.e. a C/Fortran
    function binding) is first accessed through the proxy.

    Dunder attributes are never forwarded to the library — this prevents
    introspection tools like pdoc from accidentally triggering library
    initialization (and a potential segfault from MPI_Init).
    """
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        # MMML patch: ``pycharmm.lib`` is this proxy until something imports the
        # ``pycharmm.lib`` module, which then replaces the package attribute.
        # MMML calls ``pycharmm.lib.charmm.<symbol>``, so make ``.charmm`` resolve
        # to the same library either way instead of a missing ``charmm`` symbol.
        if name == 'charmm':
            return self
        return getattr(_loader.handle, name)

    def __repr__(self):
        if is_initialized():
            return repr(_loader._lib)
        return "<pycharmm CHARMM library (not yet initialized)>"


lib = _LazyLib()


def get_lib_path():
    """get the path of the charmm shared library as a string

       Returns
       -------
       string
           the path of the charmm shared library
    """
    return _loader.charmm_lib_name[:]


def print_lib_path():
    """print the path of the charmm shared library
    """
    print(get_lib_path())
