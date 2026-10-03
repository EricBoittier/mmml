"""Configure CHARMM's fixed-size array dimensions (DIMENS)

This module provides classes and an instance to set CHARMM array dimensions
PRIOR to the initialization of the CHARMM library. These settings are
communicated to CHARMM when it's first initialized (lazily).

Corresponds to CHARMM command `DIMENSion`.
See CHARMM documentation [dimens](<https://academiccharmm.org/documentation/version/latest/dimens>)
for more information.

Usage:
>>> import pycharmm
>>> pycharmm.dimens.set_chsize(500000) # Set before any CHARMM operation
>>> # ... other pycharmm operations that will trigger CHARMM initialization

Classes:
    Dim: Utility class to store and manage a single dimension size.
    Dimens: Manages all configurable CHARMM dimensions.

Instance:
    dimens: A global instance of the Dimens class, accessible via
            `pycharmm.dimens` after `import pycharmm`.
"""

import copy
import ctypes
import sys


class Dim:
    _value : int
    _changed : bool

    def __init__(self, default=1):
        """a utility class to store a single size for a dimens array of charmm

        Parameters
        ----------
        default : int
            A nonzero positive integer, the default size for a dimens array
        """
        self._validate_value(default)
        self._value = default
        self._changed = False

    def set(self, new_value: int) -> int:
        """set the value of this particular Dim

        Parameters
        ----------
        new_value : int
            a nonzero positive integer, a new size for the dimens array

        Returns
        -------
        old_value : integer
            previous size for the dimens array
        """
        self._validate_value(new_value)
        old_value = self._value
        if (new_value != old_value):
            self._value = new_value
            self._changed = True

        return old_value

    def get(self) -> int:
        """get the current value of this particular Dim

        Returns
        -------
        val : integer
            a copy of the current size for this dimens array
        """
        val = copy.deepcopy(self._value)
        return val

    def has_changed(self) -> bool:
        """Has this Dim every had its value set after initialization?

        Returns
        -------
        q : bool
            True if changed after init, False otherwise
        """
        q = copy.deepcopy(self._changed)
        return q

    def _validate_value(self, new_value):
        if not (isinstance(new_value, int) and (new_value > 0)):
            raise ValueError("Dim default value must be " +
                             " an integer greater than zero.")

        return new_value


class Dimens:
    _chsize : Dim
    _maxa : Dim
    _maxb : Dim
    _maxt : Dim
    _maxp : Dim
    _maximp : Dim
    _maxnb : Dim
    _maxpad : Dim
    _maxres : Dim
    _maxseg : Dim
    _maxcrt : Dim
    _maxshk : Dim
    _maxaim : Dim
    _maxgrp : Dim
    _maxnbf : Dim
    _maxitc : Dim
    _iatbmx : Dim
    _maxpar : Dim

    def __init__(self):
        """a utility class to store sizes for dimens arrays of charmm

        With lazy loading, importing pycharmm does NOT initialize CHARMM, so
        `pycharmm.dimens` may be manipulated any time before the first CHARMM
        call (which triggers initialization). After CHARMM initializes, a
        Dimens object has no further effect.
        """
        default_chsize = 360720
        self._chsize = Dim(default_chsize)
        self._maxa = Dim(default_chsize)
        self._maxb = Dim(default_chsize)
        self._maxt = Dim(2 * default_chsize)
        self._maxp = Dim(3 * default_chsize)
        self._maximp = Dim(default_chsize // 2)
        self._maxnb = Dim(default_chsize // 4)
        self._maxpad = Dim(default_chsize)
        self._maxres = Dim(default_chsize // 3)
        self._maxseg = Dim(default_chsize // 8)
        self._maxcrt = Dim(default_chsize // 3)
        self._maxshk = Dim(default_chsize)
        self._maxaim = Dim(2 * default_chsize)
        self._maxgrp = Dim(2 * default_chsize // 3)
        self._maxnbf = Dim(default_chsize // 360)
        self._maxitc = Dim(default_chsize // 360)
        # IATBMX: max bonds per atom (chsizes dimension; CHARMM default 32
        # with BLOCK else 8). MAXPAR: command-parser token-table size (cmdpar,
        # NOT a chsizes dimension; CHARMM default 10240).
        self._iatbmx = Dim(32)
        self._maxpar = Dim(10240)

    @property
    def chsize(self) -> int:
        return self._chsize.get()

    @property
    def maxa(self) -> int:
        return self._maxa.get()

    @property
    def maxb(self) -> int:
        return self._maxb.get()

    @property
    def maxt(self) -> int:
        return self._maxt.get()

    @property
    def maxp(self) -> int:
        return self._maxp.get()

    @property
    def maximp(self) -> int:
        return self._maximp.get()

    @property
    def maxnb(self) -> int:
        return self._maxnb.get()

    @property
    def maxpad(self) -> int:
        return self._maxpad.get()

    @property
    def maxres(self) -> int:
        return self._maxres.get()

    @property
    def maxseg(self) -> int:
        return self._maxseg.get()

    @property
    def maxcrt(self) -> int:
        return self._maxcrt.get()

    @property
    def maxshk(self) -> int:
        return self._maxshk.get()

    @property
    def maxaim(self) -> int:
        return self._maxaim.get()

    @property
    def maxgrp(self) -> int:
        return self._maxgrp.get()

    @property
    def maxnbf(self) -> int:
        return self._maxnbf.get()

    @property
    def maxitc(self) -> int:
        return self._maxitc.get()

    @property
    def iatbmx(self) -> int:
        return self._iatbmx.get()

    @property
    def maxpar(self) -> int:
        return self._maxpar.get()

    def set_chsize(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'chsize' cannot be changed. Current value {self._chsize.get()} remains.", file=sys.stderr)
            return self._chsize.get()
        old_value = self._chsize.set(new_value)
        return old_value

    def set_maxa(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxa' cannot be changed. Current value {self._maxa.get()} remains.", file=sys.stderr)
            return self._maxa.get()
        old_value = self._maxa.set(new_value)
        return old_value

    def set_maxb(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxb' cannot be changed. Current value {self._maxb.get()} remains.", file=sys.stderr)
            return self._maxb.get()
        old_value = self._maxb.set(new_value)
        return old_value

    def set_maxt(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxt' cannot be changed. Current value {self._maxt.get()} remains.", file=sys.stderr)
            return self._maxt.get()
        old_value = self._maxt.set(new_value)
        return old_value

    def set_maxp(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxp' cannot be changed. Current value {self._maxp.get()} remains.", file=sys.stderr)
            return self._maxp.get()
        old_value = self._maxp.set(new_value)
        return old_value

    def set_maximp(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maximp' cannot be changed. Current value {self._maximp.get()} remains.", file=sys.stderr)
            return self._maximp.get()
        old_value = self._maximp.set(new_value)
        return old_value

    def set_maxnb(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxnb' cannot be changed. Current value {self._maxnb.get()} remains.", file=sys.stderr)
            return self._maxnb.get()
        old_value = self._maxnb.set(new_value)
        return old_value

    def set_maxpad(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxpad' cannot be changed. Current value {self._maxpad.get()} remains.", file=sys.stderr)
            return self._maxpad.get()
        old_value = self._maxpad.set(new_value)
        return old_value

    def set_maxres(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxres' cannot be changed. Current value {self._maxres.get()} remains.", file=sys.stderr)
            return self._maxres.get()
        old_value = self._maxres.set(new_value)
        return old_value

    def set_maxseg(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxseg' cannot be changed. Current value {self._maxseg.get()} remains.", file=sys.stderr)
            return self._maxseg.get()
        old_value = self._maxseg.set(new_value)
        return old_value

    def set_maxcrt(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxcrt' cannot be changed. Current value {self._maxcrt.get()} remains.", file=sys.stderr)
            return self._maxcrt.get()
        old_value = self._maxcrt.set(new_value)
        return old_value

    def set_maxshk(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxshk' cannot be changed. Current value {self._maxshk.get()} remains.", file=sys.stderr)
            return self._maxshk.get()
        old_value = self._maxshk.set(new_value)
        return old_value

    def set_maxaim(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxaim' cannot be changed. Current value {self._maxaim.get()} remains.", file=sys.stderr)
            return self._maxaim.get()
        old_value = self._maxaim.set(new_value)
        return old_value

    def set_maxgrp(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxgrp' cannot be changed. Current value {self._maxgrp.get()} remains.", file=sys.stderr)
            return self._maxgrp.get()
        old_value = self._maxgrp.set(new_value)
        return old_value

    def set_maxnbf(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxnbf' cannot be changed. Current value {self._maxnbf.get()} remains.", file=sys.stderr)
            return self._maxnbf.get()
        old_value = self._maxnbf.set(new_value)
        return old_value

    def set_maxitc(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'maxitc' cannot be changed. Current value {self._maxitc.get()} remains.", file=sys.stderr)
            return self._maxitc.get()
        old_value = self._maxitc.set(new_value)
        return old_value

    def set_iatbmx(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. Dimension 'iatbmx' cannot be changed. Current value {self._iatbmx.get()} remains.", file=sys.stderr)
            return self._iatbmx.get()
        old_value = self._iatbmx.set(new_value)
        return old_value

    def set_maxpar(self, new_value) -> int:
        from pycharmm.loader import is_initialized
        if is_initialized():
            print(f"Warning: CHARMM is already initialized. MAXPAR cannot be changed. Current value {self._maxpar.get()} remains.", file=sys.stderr)
            return self._maxpar.get()
        old_value = self._maxpar.set(new_value)
        return old_value

    def communicate_dimens(self, lib):
        """send any changed sizes to the charmm shared library

        This will be called in the init method of the CharmmLib class.
        Once the library is initialized, updating dimens will not be
        possible.

        Parameters
        ----------
        lib :
            a charmm shared library created using ctypes
            This charmm will have its dimens updated.
        """
        if self._chsize.has_changed():
            new_dim = ctypes.c_int(self._chsize.get())
            lib.dimens_set_chsize(ctypes.byref(new_dim))

        if self._maxa.has_changed():
            new_dim = ctypes.c_int(self._maxa.get())
            lib.dimens_set_maxa(ctypes.byref(new_dim))

        if self._maxb.has_changed():
            new_dim = ctypes.c_int(self._maxb.get())
            lib.dimens_set_maxb(ctypes.byref(new_dim))

        if self._maxt.has_changed():
            new_dim = ctypes.c_int(self._maxt.get())
            lib.dimens_set_maxt(ctypes.byref(new_dim))

        if self._maxp.has_changed():
            new_dim = ctypes.c_int(self._maxp.get())
            lib.dimens_set_maxp(ctypes.byref(new_dim))

        if self._maximp.has_changed():
            new_dim = ctypes.c_int(self._maximp.get())
            lib.dimens_set_maximp(ctypes.byref(new_dim))

        if self._maxnb.has_changed():
            new_dim = ctypes.c_int(self._maxnb.get())
            lib.dimens_set_maxnb(ctypes.byref(new_dim))

        if self._maxpad.has_changed():
            new_dim = ctypes.c_int(self._maxpad.get())
            lib.dimens_set_maxpad(ctypes.byref(new_dim))

        if self._maxres.has_changed():
            new_dim = ctypes.c_int(self._maxres.get())
            lib.dimens_set_maxres(ctypes.byref(new_dim))

        if self._maxseg.has_changed():
            new_dim = ctypes.c_int(self._maxseg.get())
            lib.dimens_set_maxseg(ctypes.byref(new_dim))

        if self._maxcrt.has_changed():
            new_dim = ctypes.c_int(self._maxcrt.get())
            lib.dimens_set_maxcrt(ctypes.byref(new_dim))

        if self._maxshk.has_changed():
            new_dim = ctypes.c_int(self._maxshk.get())
            lib.dimens_set_maxshk(ctypes.byref(new_dim))

        if self._maxaim.has_changed():
            new_dim = ctypes.c_int(self._maxaim.get())
            lib.dimens_set_maxaim(ctypes.byref(new_dim))

        if self._maxgrp.has_changed():
            new_dim = ctypes.c_int(self._maxgrp.get())
            lib.dimens_set_maxgrp(ctypes.byref(new_dim))

        if self._maxnbf.has_changed():
            new_dim = ctypes.c_int(self._maxnbf.get())
            lib.dimens_set_maxnbf(ctypes.byref(new_dim))

        if self._maxitc.has_changed():
            new_dim = ctypes.c_int(self._maxitc.get())
            lib.dimens_set_maxitc(ctypes.byref(new_dim))

        if self._iatbmx.has_changed():
            new_dim = ctypes.c_int(self._iatbmx.get())
            lib.dimens_set_iatbmx(ctypes.byref(new_dim))

        # MAXPAR is a cmdpar table size, not a chsizes dimension -> its own
        # C entry point (cmdpar_set_maxpar), not dimens_set_*.
        if self._maxpar.has_changed():
            new_dim = ctypes.c_int(self._maxpar.get())
            lib.cmdpar_set_maxpar(ctypes.byref(new_dim))

    def show(self, lib):
        """print out the sizes of all the dimens arrays from charmm
        """
        lib.dimens_print()

    def show_config(self):
        """Show the currently configured dimension values from the Python-side object."""
        print(f"Configured CHSIZE: {self._chsize.get()}")
        print(f"Configured MAXA: {self._maxa.get()}")
        print(f"Configured MAXB: {self._maxb.get()}")
        print(f"Configured MAXT: {self._maxt.get()}")
        print(f"Configured MAXP: {self._maxp.get()}")
        print(f"Configured MAXIMP: {self._maximp.get()}")
        print(f"Configured MAXNB: {self._maxnb.get()}")
        print(f"Configured MAXPAD: {self._maxpad.get()}")
        print(f"Configured MAXRES: {self._maxres.get()}")
        print(f"Configured MAXSEG: {self._maxseg.get()}")
        print(f"Configured MAXCRT: {self._maxcrt.get()}")
        print(f"Configured MAXSHK: {self._maxshk.get()}")
        print(f"Configured MAXAIM: {self._maxaim.get()}")
        print(f"Configured MAXGRP: {self._maxgrp.get()}")
        print(f"Configured MAXNBF: {self._maxnbf.get()}")
        print(f"Configured MAXITC: {self._maxitc.get()}")


dimens = Dimens() 