"""Time series analysis for CHARMM trajectories.

Provides ``Series`` (numpy wrapper with metadata and MANTIM-inspired
transforms) and ``Trajectory`` (DCD reader with geometry extraction).
Geometry math is performed by Fortran ``bind(c)`` functions in
``api_correl.F90`` for speed; transforms are pure numpy.

The transform methods on :class:`Series` mirror CHARMM's MANTIM
(MANipulate TIMe series) commands, using the same names where possible
so that users familiar with CHARMM scripting can transfer their
knowledge directly.

Examples
========
>>> from pycharmm.correl import Trajectory
>>> traj = Trajectory("dynamics.dcd")
>>> phi, psi = traj.ramachandran(resid=5)
>>> phi_smooth = phi.cont().movi(10)          # unwrap, then smooth
>>> dphi = phi_smooth.deri(dt=0.002)          # angular velocity
>>> counts, edges = phi.hist(bins=72, range=(-180, 180))
"""

import ctypes
import warnings

import numpy as np

import pycharmm.coor as coor
from pycharmm.loader import lib
import pycharmm.lingo as lingo


def _register_dcd_signatures():
    """Register ctypes signatures for the direct DCD reader.

    Called lazily from _read_dcd_direct rather than at import time:
    touching ``lib`` triggers CHARMM library initialization (see
    pycharmm.loader._LazyLib), so registering at module scope would
    boot the Fortran runtime on every ``import pycharmm``.  All three
    functions return integer status codes; the arguments are pointers.
    """
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_dbl_p = ctypes.POINTER(ctypes.c_double)
    lib.correl_dcd_open.restype = ctypes.c_int
    lib.correl_dcd_open.argtypes = [
        ctypes.c_char_p, c_int_p, c_int_p, c_dbl_p, c_int_p, c_int_p]
    lib.correl_dcd_read_frame.restype = ctypes.c_int
    lib.correl_dcd_read_frame.argtypes = [c_dbl_p, c_dbl_p, c_dbl_p]
    lib.correl_dcd_close.restype = ctypes.c_int
    lib.correl_dcd_close.argtypes = []


def _resolve_step_selection(begin, stop, skip, nsavc, first_step, nframes):
    """Normalize a begin/stop/skip request against a DCD header.

    ``begin``, ``stop``, ``skip`` are step numbers (``None`` meaning "read
    the whole trajectory").  ``nsavc`` and ``first_step`` must already be
    positive (callers clamp them).  ``nframes`` is the stored frame count
    from the header, or ``<= 0`` if the header does not record one.

    This is the single source of truth for the step->frame grid: it owns
    both the last-stored-step derivation and the begin/stop/skip clamping,
    so the two Trajectory readers select the same frames.  Returns
    ``(r_begin, r_stop, r_skip)`` in step units:

    * ``r_begin`` is rounded up to a stored step (``first_step + k*nsavc``)
      and never falls below ``first_step`` (READCV aborts on begin < ISTEP1);
    * ``r_skip`` is a positive multiple of ``nsavc`` (READCV requires it);
    * ``r_stop`` is clamped to the last stored step; it is ``None`` when the
      frame count is unknown and no explicit ``stop`` was given, meaning
      "read to end of file".

    The lingo path uses the step values directly, the direct path converts
    them to frame indices via ``first_step``/``nsavc``.
    """
    last_step = first_step + (nframes - 1) * nsavc if nframes > 0 else None

    if skip is None:
        r_skip = nsavc
    else:
        r_skip = max(1, -(-int(skip) // nsavc)) * nsavc

    if begin is None:
        r_begin = first_step
    else:
        r_begin = max(first_step, int(begin))
        offset = r_begin - first_step
        if offset % nsavc != 0:  # round up to the next stored step
            r_begin = first_step + (offset // nsavc + 1) * nsavc

    if stop is None or stop <= 0:
        r_stop = last_step
    elif last_step is not None:
        r_stop = min(int(stop), last_step)
    else:
        r_stop = int(stop)

    return r_begin, r_stop, r_skip


class Series:
    """A named time series with units, wrapping a numpy array.

    Provides statistics, arithmetic operators, and a comprehensive set
    of MANTIM-inspired transforms that return new ``Series`` objects,
    enabling fluent method chaining::

        result = series.dave().movi(20).deri(dt=0.001)

    Parameters
    ----------
    values : array_like
        The time series data.
    name : str
        Descriptive name for the series.
    units : str
        Physical units (e.g. "Angstrom", "degrees").
    dt : float
        Time step between successive points (picoseconds).
        Used by :meth:`deri` and :meth:`inte` for proper scaling.
        Default 1.0 (dimensionless index units).
    """

    def __init__(self, values, name="", units="", dt=1.0):
        self._values = np.asarray(values, dtype=np.float64)
        self.name = name
        self.units = units
        self.dt = float(dt)

    def _new(self, values, name=None, units=None, dt=None):
        """Create a new Series inheriting metadata from this one."""
        return Series(
            values,
            name=name if name is not None else self.name,
            units=units if units is not None else self.units,
            dt=dt if dt is not None else self.dt)

    @property
    def values(self):
        """The underlying numpy array (float64)."""
        return self._values

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        n = len(self._values)
        parts = [f"Series(n={n}"]
        if self.name:
            parts.append(f"name='{self.name}'")
        if self.units:
            parts.append(f"units='{self.units}'")
        parts.append(f"mean={self.mean():.4f}")
        return ", ".join(parts) + ")"

    def __getitem__(self, key):
        return self._values[key]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def mean(self):
        """Arithmetic mean of the series."""
        return float(np.mean(self._values))

    def std(self):
        """Standard deviation (population, ddof=0)."""
        return float(np.std(self._values))

    def min(self):
        """Minimum value."""
        return float(np.min(self._values))

    def max(self):
        """Maximum value."""
        return float(np.max(self._values))

    def histogram(self, bins=50, range=None):
        """Compute histogram of the series values.

        Returns
        -------
        counts : ndarray
            Bin counts.
        edges : ndarray
            Bin edges (length = len(counts) + 1).
        """
        return np.histogram(self._values, bins=bins, range=range)

    # ------------------------------------------------------------------
    # Arithmetic operators (return new Series)
    # ------------------------------------------------------------------

    def _binop(self, other, op, symbol):
        if isinstance(other, Series):
            result = op(self._values, other._values)
            name = f"({self.name} {symbol} {other.name})" if self.name or other.name else ""
        else:
            result = op(self._values, other)
            name = self.name
        units = self.units if not isinstance(other, Series) else ""
        return self._new(result, name=name, units=units)

    def __add__(self, other):
        return self._binop(other, np.add, "+")

    def __sub__(self, other):
        return self._binop(other, np.subtract, "-")

    def __mul__(self, other):
        return self._binop(other, np.multiply, "*")

    def __truediv__(self, other):
        return self._binop(other, np.true_divide, "/")

    # ------------------------------------------------------------------
    # MANTIM transforms — all return new Series for chaining
    # ------------------------------------------------------------------

    # --- Centering / normalization ---

    def dave(self):
        """Subtract the mean (detrend).  MANTIM: ``DAVE``.

        Returns a zero-mean series: ``Q(t) - <Q>``.
        """
        return self._new(self._values - np.mean(self._values),
                         name=f"dave({self.name})")

    def dini(self):
        """Subtract the initial value.  MANTIM: ``DINI``.

        Returns ``Q(t) - Q(0)``.
        """
        return self._new(self._values - self._values[0],
                         name=f"dini({self.name})")

    def dmin(self):
        """Subtract the minimum value.  MANTIM: ``DMIN``.

        Shifts the series so that its minimum is zero.
        """
        return self._new(self._values - np.min(self._values),
                         name=f"dmin({self.name})")

    def divf(self):
        """Divide by the first value.  MANTIM: ``DIVF``.

        Normalizes so that the series starts at 1.0.

        Raises
        ------
        ValueError
            If the first value is zero.
        """
        v0 = self._values[0]
        if v0 == 0.0:
            raise ValueError("divf: first value is zero")
        return self._new(self._values / v0,
                         name=f"divf({self.name})", units="")

    def divm(self):
        """Divide by the maximum absolute value.  MANTIM: ``DIVM``.

        Normalizes the series to the range [-1, 1].

        Raises
        ------
        ValueError
            If the maximum absolute value is zero.
        """
        mx = np.max(np.abs(self._values))
        if mx == 0.0:
            raise ValueError("divm: all values are zero")
        return self._new(self._values / mx,
                         name=f"divm({self.name})", units="")

    # --- Scaling ---

    def mult(self, factor):
        """Multiply by a constant.  MANTIM: ``MULT``.

        Parameters
        ----------
        factor : float
            Scale factor.
        """
        return self._new(self._values * factor,
                         name=f"mult({self.name},{factor})")

    def shift(self, offset):
        """Add a constant offset.  MANTIM: ``SHIF``.

        Parameters
        ----------
        offset : float
            Value to add.
        """
        return self._new(self._values + offset,
                         name=f"shift({self.name},{offset})")

    def divi(self, factor):
        """Divide by a constant.  MANTIM: ``DIVI``.

        Parameters
        ----------
        factor : float
            Divisor (must be nonzero).
        """
        return self._new(self._values / factor,
                         name=f"divi({self.name},{factor})")

    # --- Math functions ---

    def log(self):
        """Natural logarithm.  MANTIM: ``LOG``.

        Values below 1e-12 are clamped to avoid -inf.
        """
        v = np.clip(self._values, 1e-12, None)
        return self._new(np.log(v), name=f"log({self.name})", units="")

    def exp(self):
        """Exponential.  MANTIM: ``EXP``.

        Values above 709 are clamped to avoid overflow.
        """
        v = np.clip(self._values, None, 709.0)
        return self._new(np.exp(v), name=f"exp({self.name})", units="")

    def sqrt(self):
        """Square root.  MANTIM: ``SQRT``.

        Negative values are clamped to zero.
        """
        return self._new(np.sqrt(np.clip(self._values, 0.0, None)),
                         name=f"sqrt({self.name})")

    def square(self):
        """Square the series.  MANTIM: ``SQUA``.

        Returns ``Q(t)^2``.
        """
        return self._new(self._values ** 2,
                         name=f"squa({self.name})")

    def abs(self):
        """Absolute value.  MANTIM: ``ABS``.
        """
        return self._new(np.abs(self._values),
                         name=f"abs({self.name})")

    def ipow(self, n):
        """Raise to an integer power.  MANTIM: ``IPOW``.

        Parameters
        ----------
        n : int
            Exponent.
        """
        return self._new(self._values ** int(n),
                         name=f"ipow({self.name},{n})")

    # --- Trigonometric (degrees) ---

    def cos(self):
        """Cosine of values assumed in degrees.  MANTIM: ``COS``.
        """
        return self._new(np.cos(np.radians(self._values)),
                         name=f"cos({self.name})", units="")

    def acos(self):
        """Arc-cosine, result in degrees.  MANTIM: ``ACOS``.

        Input is clamped to [-1, 1].
        """
        v = np.clip(self._values, -1.0, 1.0)
        return self._new(np.degrees(np.arccos(v)),
                         name=f"acos({self.name})", units="degrees")

    def cos2(self):
        """Second Legendre polynomial: ``3*cos^2(Q) - 1``.  MANTIM: ``COS2``.

        Input assumed in degrees.  Used for rotational correlation analysis.
        """
        c = np.cos(np.radians(self._values))
        return self._new(3.0 * c * c - 1.0,
                         name=f"cos2({self.name})", units="")

    # --- Filtering / smoothing ---

    def movi(self, window):
        """Moving (running) average.  MANTIM: ``MOVI``.

        Uses a causal window: point *t* is the mean of
        ``Q(max(0, t-window+1) : t+1)``.  The window grows from 1 at
        the start of the series to *window* once enough points are
        available.

        Parameters
        ----------
        window : int
            Window size (number of points).
        """
        window = int(window)
        if window < 1:
            raise ValueError("movi: window must be >= 1")
        cs = np.cumsum(self._values)
        out = np.empty_like(self._values)
        for i in range(len(self._values)):
            lo = i - window
            s = cs[i] - (cs[lo] if lo >= 0 else 0.0)
            out[i] = s / min(i + 1, window)
        return self._new(out, name=f"movi({self.name},{window})")

    def aver(self, block_size):
        """Block average — compress by averaging blocks.  MANTIM: ``AVER``.

        Divides the series into non-overlapping blocks of *block_size*
        points and replaces each block with its mean.  Partial trailing
        blocks are included.

        Parameters
        ----------
        block_size : int
            Number of points per block.
        """
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("aver: block_size must be >= 1")
        n = len(self._values)
        nblocks = (n + block_size - 1) // block_size
        out = np.empty(nblocks)
        for i in range(nblocks):
            lo = i * block_size
            hi = min(lo + block_size, n)
            out[i] = np.mean(self._values[lo:hi])
        return self._new(out, name=f"aver({self.name},{block_size})",
                         dt=self.dt * block_size)

    def deln(self, window):
        """Subtract running average (high-pass filter).  MANTIM: ``DELN``.

        Returns ``Q(t) - movi(Q, window)(t)``.

        Parameters
        ----------
        window : int
            Window size for the running average to subtract.
        """
        smoothed = self.movi(window)
        return self._new(self._values - smoothed._values,
                         name=f"deln({self.name},{window})")

    # --- Calculus ---

    def deri(self, dt=None):
        """Numerical derivative (forward difference).  MANTIM: ``DERI``.

        ``dQ/dt = (Q(t+1) - Q(t)) / dt``

        The last point is set equal to the second-to-last to preserve
        the series length.

        Parameters
        ----------
        dt : float, optional
            Time step.  Defaults to ``self.dt``.
        """
        if dt is None:
            dt = self.dt
        d = np.empty_like(self._values)
        d[:-1] = np.diff(self._values) / dt
        d[-1] = d[-2] if len(d) > 1 else 0.0
        u = f"{self.units}/ps" if self.units else ""
        return self._new(d, name=f"deri({self.name})", units=u)

    def inte(self, dt=None):
        """Cumulative numerical integral (trapezoidal).  MANTIM: ``INTE``.

        ``I(t) = integral from 0 to t of Q(tau) d(tau)``

        Parameters
        ----------
        dt : float, optional
            Time step.  Defaults to ``self.dt``.
        """
        if dt is None:
            dt = self.dt
        from scipy.integrate import cumulative_trapezoid
        c = cumulative_trapezoid(self._values, dx=dt, initial=0.0)
        u = f"{self.units}*ps" if self.units else ""
        return self._new(c, name=f"inte({self.name})", units=u)

    # --- Periodic angle utilities ---

    def cont(self, period=360.0):
        """Make a periodic series continuous.  MANTIM: ``CONT``.

        Removes jumps larger than *period/2* by adding or subtracting
        multiples of *period*.  Essential for dihedral angle time series
        before smoothing or differentiation.

        Parameters
        ----------
        period : float
            Full period of the variable (default 360 for degrees).
        """
        half = period / 2.0
        out = self._values.copy()
        for i in range(1, len(out)):
            while out[i] - out[i - 1] > half:
                out[i] -= period
            while out[i] - out[i - 1] < -half:
                out[i] += period
        return self._new(out, name=f"cont({self.name})")

    def map(self, lo=0.0, hi=360.0):
        """Map periodic values to the range [lo, hi).  MANTIM: ``MAP``.

        Parameters
        ----------
        lo : float
            Lower bound (default 0).
        hi : float
            Upper bound (default 360).
        """
        period = hi - lo
        out = ((self._values - lo) % period) + lo
        return self._new(out, name=f"map({self.name})")

    # --- Thresholding ---

    def heav(self):
        """Heaviside step function.  MANTIM: ``HEAV``.

        Returns 1.0 where ``Q(t) >= 0``, else 0.0.
        """
        return self._new(np.where(self._values >= 0.0, 1.0, 0.0),
                         name=f"heav({self.name})", units="")

    def stat(self, lo, hi):
        """Threshold to binary state.  MANTIM: ``STAT``.

        Returns 1.0 where ``lo < Q(t) < hi``, else 0.0.

        Parameters
        ----------
        lo, hi : float
            Lower and upper bounds (exclusive).
        """
        v = self._values
        out = np.where((v > lo) & (v < hi), 1.0, 0.0)
        return self._new(out, name=f"stat({self.name},{lo},{hi})",
                         units="")

    # --- Histogram transforms ---

    def prob(self, bins=50):
        """Auto-ranged probability histogram.  MANTIM: ``PROB``.

        Replaces the series with bin probabilities over the range
        ``[min, max]`` of the data.

        Parameters
        ----------
        bins : int
            Number of bins.

        Returns
        -------
        Series
            Probability values (sums to ~1).  The bin centers are
            available from the series index: ``lo + (i + 0.5) * bin_width``.
        """
        counts, _ = np.histogram(self._values, bins=bins)
        probs = counts / float(len(self._values))
        return self._new(probs, name=f"prob({self.name})", units="prob")

    def hist(self, bins=50, range=None):
        """Fixed-range probability histogram.  MANTIM: ``HIST``.

        Parameters
        ----------
        bins : int
            Number of bins.
        range : tuple of (float, float), optional
            ``(lo, hi)`` range.  Defaults to ``(min, max)`` of data.

        Returns
        -------
        Series
            Probability values.
        """
        counts, _ = np.histogram(self._values, bins=bins, range=range)
        probs = counts / float(len(self._values))
        return self._new(probs, name=f"hist({self.name})", units="prob")

    # --- Zeroing ---

    def zero(self):
        """Set all values to zero.  MANTIM: ``ZERO``.
        """
        return self._new(np.zeros_like(self._values),
                         name=f"zero({self.name})")

    # ------------------------------------------------------------------
    # CORFUN — correlation and spectral analysis
    # ------------------------------------------------------------------

    def acf(self, maxlag=None, normalize=True):
        """Autocorrelation function via FFT.  CORFUN: ``P1 AUTO``.

        Computes ``C(tau) = <Q(0) * Q(tau)>``, optionally normalized
        so that ``C(0) = 1``.  Uses zero-padded FFT (Wiener-Khinchin
        theorem) for O(N log N) performance.

        Parameters
        ----------
        maxlag : int, optional
            Maximum lag in frames.  Defaults to ``len(self) // 2``.
        normalize : bool
            If True (default), divide by ``C(0)`` so the result starts
            at 1.0 and decays toward 0.

        Returns
        -------
        Series
            Autocorrelation values for lags ``0, 1, ..., maxlag-1``.
            The ``dt`` attribute is preserved for time-axis reconstruction.
        """
        v = self._values - np.mean(self._values)
        n = len(v)
        if maxlag is None:
            maxlag = n // 2
        maxlag = min(maxlag, n)

        # Zero-pad to next power of 2 >= 2*n for efficient FFT
        fft_len = 1
        while fft_len < 2 * n:
            fft_len *= 2

        fv = np.fft.rfft(v, n=fft_len)
        acf_full = np.fft.irfft(fv * np.conj(fv), n=fft_len)

        # Normalize by overlap count (n - lag)
        counts = np.arange(n, n - maxlag, -1, dtype=np.float64)
        result = acf_full[:maxlag] / counts

        if normalize and result[0] != 0.0:
            result = result / result[0]

        return self._new(result, name=f"acf({self.name})", units="")

    def ccf(self, other, maxlag=None, normalize=True):
        """Cross-correlation function via FFT.  CORFUN: ``P1 CROSS``.

        Computes ``C(tau) = <A(0) * B(tau)>``, optionally normalized
        by ``sqrt(<A^2> * <B^2>)``.

        Parameters
        ----------
        other : Series
            The second time series (must have the same length).
        maxlag : int, optional
            Maximum lag in frames.  Defaults to ``len(self) // 2``.
        normalize : bool
            If True (default), normalize so that perfect correlation
            gives 1.0.

        Returns
        -------
        Series
            Cross-correlation for lags ``0, 1, ..., maxlag-1``.
        """
        a = self._values - np.mean(self._values)
        b = other._values - np.mean(other._values)
        n = len(a)
        if len(b) != n:
            raise ValueError(
                f"Series lengths must match: {n} != {len(b)}")
        if maxlag is None:
            maxlag = n // 2
        maxlag = min(maxlag, n)

        fft_len = 1
        while fft_len < 2 * n:
            fft_len *= 2

        fa = np.fft.rfft(a, n=fft_len)
        fb = np.fft.rfft(b, n=fft_len)
        ccf_full = np.fft.irfft(np.conj(fa) * fb, n=fft_len)

        counts = np.arange(n, n - maxlag, -1, dtype=np.float64)
        result = ccf_full[:maxlag] / counts

        if normalize:
            norm = np.sqrt(np.mean(a**2) * np.mean(b**2))
            if norm > 0:
                result = result / norm

        name = f"ccf({self.name},{other.name})"
        return self._new(result, name=name, units="")

    def msd(self, other=None, maxlag=None):
        """Mean squared displacement.  CORFUN: ``DIFF``.

        Computes ``MSD(tau) = <(Q(t) - Q(t+tau))^2>_t`` — the time
        average of squared displacements at lag *tau*.  Works correctly
        for both stationary (e.g. oscillators) and non-stationary
        (e.g. diffusion/random walks) processes.

        When *other* is ``None``, computes self-MSD.  When *other* is
        given, computes ``<(A(t) - B(t+tau))^2>_t``.

        Parameters
        ----------
        other : Series, optional
            Second series for cross-MSD.
        maxlag : int, optional
            Maximum lag.  Defaults to ``len(self) // 2``.

        Returns
        -------
        Series
            MSD values starting at 0 for lag 0.
        """
        a = self._values
        b = other._values if other is not None else a
        n = len(a)
        if len(b) != n:
            raise ValueError(
                f"Series lengths must match: {n} != {len(b)}")
        if maxlag is None:
            maxlag = n // 2
        maxlag = min(maxlag, n)

        result = np.empty(maxlag)
        for m in range(maxlag):
            diffs = a[:n - m] - b[m:n]
            result[m] = np.mean(diffs ** 2)

        u = f"{self.units}^2" if self.units else ""
        name = f"msd({self.name})" if other is None else \
            f"msd({self.name},{other.name})"
        return self._new(result, name=name, units=u)

    def spectrum(self, window="cosine", pad_factor=4):
        """Power spectral density via FFT of the autocorrelation.

        Mirrors CHARMM's ``SPECTR`` from ``mancor.F90``.  Computes
        the ACF, applies an optional window function, then FFTs to
        frequency space.

        Parameters
        ----------
        window : str or None
            Window function applied to the ACF before FFT.

            - ``"cosine"`` (default) — Hann-like cosine taper:
              ``(1 + cos(pi * tau / T)) / 2``.  Matches CHARMM's
              ``SWITch`` option.
            - ``"ramp"`` — linear ramp from 1 to 0.  Matches CHARMM's
              ``RAMP`` option.
            - ``None`` — no windowing (rectangular).
        pad_factor : int
            Zero-pad the ACF to ``pad_factor * len(acf)`` before FFT
            for smoother spectral interpolation (default 4).

        Returns
        -------
        freqs : ndarray
            Frequency axis in cycles per time unit (1/dt).
        power : Series
            Spectral density values.
        """
        n = len(self._values)
        maxlag = n // 2
        c = self.acf(maxlag=maxlag, normalize=False)
        acf_vals = c.values.copy()

        # Apply window
        t = np.arange(maxlag, dtype=np.float64)
        if window == "cosine":
            w = 0.5 * (1.0 + np.cos(np.pi * t / maxlag))
            acf_vals *= w
        elif window == "ramp":
            w = 1.0 - t / maxlag
            acf_vals *= w

        # Zero-pad and FFT
        nfft = maxlag * pad_factor
        # Make power of 2
        nfft_p2 = 1
        while nfft_p2 < nfft:
            nfft_p2 *= 2

        power = np.abs(np.fft.rfft(acf_vals, n=nfft_p2))
        freqs = np.fft.rfftfreq(nfft_p2, d=self.dt)

        return freqs, self._new(power, name=f"spectrum({self.name})",
                                units="", dt=freqs[1] if len(freqs) > 1 else 1.0)


def _find_atom(resid, atom_name):
    """Resolve atom name to 1-based index via CHARMM PSF.

    Parameters
    ----------
    resid : int
        Residue number (1-based).
    atom_name : str
        Atom name (e.g. "CA", "N", "C").

    Returns
    -------
    int
        1-based atom index.

    Raises
    ------
    ValueError
        If atom not found in PSF.
    """
    c_resid = ctypes.c_int(resid)
    c_name = ctypes.create_string_buffer(atom_name.encode())
    idx = lib.correl_find_atom(ctypes.byref(c_resid), c_name)
    if idx <= 0:
        raise ValueError(
            f"Atom '{atom_name}' not found in residue {resid}")
    return idx


class Trajectory:
    """A trajectory loaded into memory for analysis.

    Reads a DCD trajectory via CHARMM's trajectory reader and stores
    all frames as contiguous numpy arrays. Provides methods for
    extracting geometry time series.

    Parameters
    ----------
    dcd_path : str
        Path to DCD trajectory file.
    begin : int, optional
        First step number to read.  ``None`` (default) starts at the
        first frame stored in the file.
    stop : int, optional
        Last step number to read.  ``None`` (default) reads through the
        last frame in the file.
    skip : int, optional
        Step interval between frames to read.  ``None`` (default) reads
        every stored frame (i.e. the file's own saving interval).
    method : str
        ``"direct"`` (default) reads via Fortran binary I/O;
        any other value reads via CHARMM ``TRAJ`` commands.

    Notes
    -----
    ``begin``, ``stop``, and ``skip`` use step-number semantics,
    matching CHARMM's ``TRAJ`` and ``CORREL`` commands (see
    dynamc.info and correl.info).  For example, a DCD saved with
    ``nsavc=10`` and ``nstep=500`` contains frames at steps
    10, 20, ..., 500.  ``BEGIN`` and ``STOP`` are step numbers, and
    ``SKIP`` must be a multiple of the file's saving interval.  Leaving
    them at their defaults reads the whole trajectory, deriving these
    values from the DCD header so they always line up with stored
    frames.  To read every 10th frame starting from step 100::

        Trajectory('traj.dcd', begin=100, skip=100, stop=500)

    All frames are stored in RAM. For large trajectories, use
    begin/stop/skip to limit memory usage.
    10K frames x 50K atoms ~ 12 GB.
    """

    def __init__(self, dcd_path, begin=None, stop=None, skip=None,
                 method="direct"):
        self._natom = coor.get_natom()
        if self._natom <= 0:
            raise RuntimeError("No atoms in PSF. Load a structure first.")
        if method == "direct":
            self._x, self._y, self._z = self._read_dcd_direct(
                dcd_path, begin, stop, skip)
        else:
            self._x, self._y, self._z = self._read_dcd(
                dcd_path, begin, stop, skip)

    @classmethod
    def from_coordinates(cls, x, y, z):
        """Create a Trajectory from pre-loaded coordinate arrays.

        Parameters
        ----------
        x, y, z : array_like
            Coordinate arrays, shape (nframes, natom).

        Returns
        -------
        Trajectory
        """
        obj = cls.__new__(cls)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        if x.ndim == 2:
            obj._natom = x.shape[1]
            obj._x = np.ascontiguousarray(x.ravel())
            obj._y = np.ascontiguousarray(y.ravel())
            obj._z = np.ascontiguousarray(z.ravel())
        elif x.ndim == 1:
            raise ValueError(
                "1D arrays require natom to be known. "
                "Pass 2D arrays of shape (nframes, natom).")
        else:
            raise ValueError("Coordinate arrays must be 1D or 2D.")
        return obj

    @property
    def nframes(self):
        return len(self._x) // self._natom

    @property
    def natom(self):
        return self._natom

    def _read_dcd(self, path, begin, stop, skip):
        """Read DCD trajectory via CHARMM lingo commands.

        ``begin``, ``stop``, and ``skip`` are step numbers, passed
        directly to CHARMM's ``TRAJ FIRST`` command.
        """
        from pycharmm import CharmmFile
        saved_pos = coor.get_positions()
        frames_x = []
        frames_y = []
        frames_z = []

        dcd = CharmmFile(file_name=path, file_unit=-1,
                         read_only=True, formatted=False)
        unit = dcd.file_unit

        # TRAJ QUERY reads the DCD header and sets substitution
        # parameters accessible via get_energy_value (not
        # get_charmm_variable): NFILE, START, SKIP, NSTEP, NDEGF, DELTA.
        lingo.charmm_script(f'traj query unit {unit}')
        nfile = int(lingo.get_energy_value('NFILE') or 0)
        nsavc = int(lingo.get_energy_value('SKIP') or 1)
        if nsavc < 1:
            nsavc = 1
        # First stored step (ISTEP1); frames live at first_step,
        # first_step+nsavc, ...  Falls back to nsavc for a run started
        # from step 0.
        first_step = int(lingo.get_energy_value('START') or nsavc)
        if first_step < nsavc:
            first_step = nsavc

        # The lingo loop is driven by a computed frame count (TRAJ READ
        # does not report EOF via IOSTAT), so it needs a known NFILE.  A
        # header without one (some externally written DCDs) cannot be
        # paged safely here; the direct reader handles those by reading to
        # EOF, so point the caller there rather than risk over-reading.
        if nfile <= 0:
            dcd.close()
            raise RuntimeError(
                f"{path}: DCD header reports no frame count (NFILE=0); "
                "the lingo trajectory reader cannot page it. Use "
                "method='direct'.")

        r_begin, r_stop, r_skip = _resolve_step_selection(
            begin, stop, skip, nsavc, first_step, nfile)

        # Compute expected frame count rather than relying on IOSTAT,
        # which is not set by TRAJ READ (only by OPEN).  r_stop is not
        # None because nfile > 0 here.
        expected = max(0, (r_stop - r_begin) // r_skip + 1)

        lingo.charmm_script(
            f'traj first {unit} nunit 1 begin {r_begin} '
            f'stop {r_stop} skip {r_skip}')

        for _ in range(expected):
            lingo.charmm_script('traj read')
            pos = coor.get_positions()
            frames_x.append(np.array(pos['x'].tolist(), dtype=np.float64))
            frames_y.append(np.array(pos['y'].tolist(), dtype=np.float64))
            frames_z.append(np.array(pos['z'].tolist(), dtype=np.float64))

        dcd.close()
        coor.set_positions(saved_pos)

        if len(frames_x) == 0:
            raise RuntimeError(f"No frames read from {path}")

        x = np.ascontiguousarray(np.concatenate(frames_x))
        y = np.ascontiguousarray(np.concatenate(frames_y))
        z = np.ascontiguousarray(np.concatenate(frames_z))
        return x, y, z

    def _read_dcd_direct(self, path, begin, stop, skip):
        """Read DCD trajectory via direct Fortran binary I/O.

        Bypasses the CHARMM lingo command layer for 10-100x faster
        reading.  Calls ``correl_dcd_open``, ``correl_dcd_read_frame``,
        and ``correl_dcd_close`` from ``api_correl.F90``.

        ``begin``, ``stop``, and ``skip`` are step numbers; they are
        converted to frame indices using *nsavc* from the DCD header.
        """
        _register_dcd_signatures()
        c_path = ctypes.create_string_buffer(path.encode())
        c_natom = ctypes.c_int()
        c_nframes = ctypes.c_int()
        c_delta = ctypes.c_double()
        c_skip = ctypes.c_int()
        c_istep1 = ctypes.c_int()

        rc = lib.correl_dcd_open(
            c_path,
            ctypes.byref(c_natom), ctypes.byref(c_nframes),
            ctypes.byref(c_delta), ctypes.byref(c_skip),
            ctypes.byref(c_istep1))
        if rc == -2:
            # Trajectory variant the direct reader does not handle (fixed
            # atoms or 4D).  Fall back to the lingo path, which uses
            # CHARMM's full READCV logic, so method='direct' still returns
            # correct data instead of silently misreading.
            warnings.warn(
                f"DCD {path} uses fixed atoms or 4D coordinates; the "
                "direct reader cannot handle these, falling back to the "
                "lingo trajectory reader.",
                RuntimeWarning, stacklevel=2)
            return self._read_dcd(path, begin, stop, skip)
        if rc != 1:
            raise RuntimeError(f"Failed to open DCD file: {path}")

        file_natom = c_natom.value
        nsavc = c_skip.value
        if nsavc < 1:
            nsavc = 1
        first_step = c_istep1.value
        if first_step < nsavc:
            first_step = nsavc
        nframes = c_nframes.value

        if file_natom != self._natom:
            lib.correl_dcd_close()
            raise RuntimeError(
                f"DCD atom count ({file_natom}) != PSF atom count "
                f"({self._natom})")

        # Resolve the selection in step units through the shared helper,
        # anchored on the file's first stored step (ISTEP1) so this path
        # agrees with the lingo path even for restart trajectories, then
        # convert to 1-based frame indices.  f_stop is None (read to EOF)
        # only when the helper leaves r_stop unbounded (unknown frame
        # count and no explicit stop); an empty selection instead yields
        # f_stop <= 0, so the loop stops immediately like the lingo path.
        r_begin, r_stop, r_skip = _resolve_step_selection(
            begin, stop, skip, nsavc, first_step, nframes)
        f_begin = (r_begin - first_step) // nsavc + 1
        f_skip = r_skip // nsavc
        f_stop = None if r_stop is None else (r_stop - first_step) // nsavc + 1

        frames_x = []
        frames_y = []
        frames_z = []
        frame_x = np.empty(file_natom, dtype=np.float64)
        frame_y = np.empty(file_natom, dtype=np.float64)
        frame_z = np.empty(file_natom, dtype=np.float64)

        px = frame_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        py = frame_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pz = frame_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        frame_idx = 0
        reached_eof = False
        while True:
            rc = lib.correl_dcd_read_frame(px, py, pz)
            if rc == 0:  # EOF
                reached_eof = True
                break
            if rc < 0:
                # The direct reader lost frame alignment (an on-disk frame
                # layout it does not model -- e.g. a per-frame record it did
                # not skip).  Rather than fail, fall back to the lingo reader,
                # which uses CHARMM's full READCV logic, so method='direct'
                # still returns correct data.  This mirrors the -2 (open-time)
                # fallback for fixed-atom/4D trajectories.
                lib.correl_dcd_close()
                warnings.warn(
                    f"Direct DCD reader lost frame alignment in {path} "
                    f"(near frame {frame_idx + 1}); falling back to the lingo "
                    "trajectory reader.",
                    RuntimeWarning, stacklevel=2)
                return self._read_dcd(path, begin, stop, skip)

            frame_idx += 1
            if f_stop is not None and frame_idx > f_stop:
                break
            if frame_idx < f_begin:
                continue
            if (frame_idx - f_begin) % f_skip != 0:
                continue

            frames_x.append(frame_x.copy())
            frames_y.append(frame_y.copy())
            frames_z.append(frame_z.copy())

        lib.correl_dcd_close()

        # If a full read to EOF returned fewer frames than the header
        # advertised, the reader silently skipped frames -- most likely an
        # undetectable per-frame record (e.g. FLUCQ) that has no ICNTRL flag.
        # Warn so the mismatch is visible instead of silently truncating.
        if reached_eof and nframes > 0 and frame_idx < nframes:
            warnings.warn(
                f"Direct DCD reader read {frame_idx} frames from {path} but "
                f"the header advertises {nframes}; the file may use a frame "
                "layout the direct reader does not model. Consider method=''.",
                RuntimeWarning, stacklevel=2)

        if len(frames_x) == 0:
            raise RuntimeError(f"No frames read from {path}")

        x = np.ascontiguousarray(np.concatenate(frames_x))
        y = np.ascontiguousarray(np.concatenate(frames_y))
        z = np.ascontiguousarray(np.concatenate(frames_z))
        return x, y, z

    def _ptr(self, arr):
        """Get ctypes double pointer for a numpy array."""
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def distance(self, i, j):
        """Compute distance time series between atoms i and j.

        Parameters
        ----------
        i, j : int
            Atom indices (1-based).

        Returns
        -------
        Series
            Distances in Angstroms.
        """
        out = np.empty(self.nframes, dtype=np.float64)
        lib.correl_distance_series(
            self._ptr(self._x), self._ptr(self._y), self._ptr(self._z),
            ctypes.c_int(self._natom), ctypes.c_int(self.nframes),
            ctypes.c_int(i), ctypes.c_int(j),
            self._ptr(out))
        return Series(out, name=f"dist({i},{j})", units="Angstrom")

    def angle(self, i, j, k):
        """Compute angle time series for atoms i-j-k.

        Parameters
        ----------
        i, j, k : int
            Atom indices (1-based). j is the vertex atom.

        Returns
        -------
        Series
            Angles in degrees.
        """
        out = np.empty(self.nframes, dtype=np.float64)
        lib.correl_angle_series(
            self._ptr(self._x), self._ptr(self._y), self._ptr(self._z),
            ctypes.c_int(self._natom), ctypes.c_int(self.nframes),
            ctypes.c_int(i), ctypes.c_int(j), ctypes.c_int(k),
            self._ptr(out))
        return Series(out, name=f"angle({i},{j},{k})", units="degrees")

    def dihedral(self, i, j, k, l):
        """Compute dihedral angle time series for atoms i-j-k-l.

        Parameters
        ----------
        i, j, k, l : int
            Atom indices (1-based).

        Returns
        -------
        Series
            Dihedral angles in degrees (-180..180).
        """
        out = np.empty(self.nframes, dtype=np.float64)
        lib.correl_dihedral_series(
            self._ptr(self._x), self._ptr(self._y), self._ptr(self._z),
            ctypes.c_int(self._natom), ctypes.c_int(self.nframes),
            ctypes.c_int(i), ctypes.c_int(j), ctypes.c_int(k),
            ctypes.c_int(l), self._ptr(out))
        return Series(out, name=f"dihe({i},{j},{k},{l})", units="degrees")

    def phi(self, resid):
        """Compute backbone phi dihedral for a residue.

        phi = C(resid-1) - N(resid) - CA(resid) - C(resid)

        Parameters
        ----------
        resid : int
            Residue number (1-based). Must be > 1.

        Returns
        -------
        Series
            Phi angles in degrees.
        """
        if resid <= 1:
            raise ValueError(
                f"phi undefined for resid {resid}: "
                "needs C atom from previous residue")
        i = _find_atom(resid - 1, "C")
        j = _find_atom(resid, "N")
        k = _find_atom(resid, "CA")
        l = _find_atom(resid, "C")
        s = self.dihedral(i, j, k, l)
        s.name = f"phi({resid})"
        return s

    def psi(self, resid):
        """Compute backbone psi dihedral for a residue.

        psi = N(resid) - CA(resid) - C(resid) - N(resid+1)

        Parameters
        ----------
        resid : int
            Residue number (1-based).

        Returns
        -------
        Series
            Psi angles in degrees.
        """
        i = _find_atom(resid, "N")
        j = _find_atom(resid, "CA")
        k = _find_atom(resid, "C")
        l = _find_atom(resid + 1, "N")
        s = self.dihedral(i, j, k, l)
        s.name = f"psi({resid})"
        return s

    def omega(self, resid):
        """Compute backbone omega dihedral for a residue.

        omega = CA(resid) - C(resid) - N(resid+1) - CA(resid+1)

        Parameters
        ----------
        resid : int
            Residue number (1-based).

        Returns
        -------
        Series
            Omega angles in degrees.
        """
        i = _find_atom(resid, "CA")
        j = _find_atom(resid, "C")
        k = _find_atom(resid + 1, "N")
        l = _find_atom(resid + 1, "CA")
        s = self.dihedral(i, j, k, l)
        s.name = f"omega({resid})"
        return s

    def ramachandran(self, resid):
        """Compute phi and psi for a residue.

        Parameters
        ----------
        resid : int
            Residue number (1-based).

        Returns
        -------
        tuple of (Series, Series)
            (phi, psi) angle series in degrees.
        """
        return self.phi(resid), self.psi(resid)

    def energy_series(self, terms=None):
        """Compute energy decomposition for every frame.

        Loads each frame's coordinates into CHARMM, evaluates the
        energy, and collects the requested terms as :class:`Series`
        objects.

        Parameters
        ----------
        terms : list of str, optional
            Energy term names to collect (e.g. ``["BOND", "ANGL",
            "VDW", "ELEC"]``).  Defaults to all active terms plus
            ``ENER`` (total potential energy).

        Returns
        -------
        dict of str -> Series
            Mapping from term name to time series of that energy
            component in kcal/mol.

        Notes
        -----
        This modifies the current CHARMM coordinate state.  The
        original coordinates are restored after the computation.
        Each frame requires a full energy evaluation, so this is
        slower than geometry extraction.
        """
        import pycharmm.energy as energy

        saved_pos = coor.get_positions()

        # Determine which terms to collect
        if terms is None:
            lingo.charmm_script("energy")
            term_names = energy.get_term_names()
            statuses = energy.get_term_statuses()
            terms = [n for n, s in zip(term_names, statuses) if s]
            terms = ["ENER"] + terms

        collectors = {t: [] for t in terms}
        natom = self._natom

        for f in range(self.nframes):
            off = f * natom
            frame_x = self._x[off:off + natom]
            frame_y = self._y[off:off + natom]
            frame_z = self._z[off:off + natom]

            import pandas
            pos = pandas.DataFrame({
                'x': frame_x.tolist(),
                'y': frame_y.tolist(),
                'z': frame_z.tolist()})
            coor.set_positions(pos)
            lingo.charmm_script("energy")

            for t in terms:
                if t == "ENER":
                    collectors[t].append(
                        energy.get_property_by_name("ENER"))
                else:
                    try:
                        collectors[t].append(
                            energy.get_term_by_name(t))
                    except ValueError:
                        collectors[t].append(0.0)

        coor.set_positions(saved_pos)

        result = {}
        for t in terms:
            result[t] = Series(collectors[t], name=t, units="kcal/mol",
                               dt=self.dt if hasattr(self, 'dt') else 1.0)
        return result


class Collector:
    """Collect time series data during live dynamics.

    Runs molecular dynamics in segments, extracting geometry and energy
    data between segments.  This avoids modifying the CHARMM dynamics
    loop while providing real-time data collection.

    Parameters
    ----------
    nsteps : int
        Total number of dynamics steps to run.
    nsavc : int
        Steps between data collection points (collection frequency).
    monitors : list of dict
        Each dict describes a quantity to monitor.  Supported types:

        - ``{"type": "distance", "i": int, "j": int}``
        - ``{"type": "angle", "i": int, "j": int, "k": int}``
        - ``{"type": "dihedral", "i": int, "j": int, "k": int, "l": int}``
        - ``{"type": "phi", "resid": int}``
        - ``{"type": "psi", "resid": int}``
        - ``{"type": "energy", "term": str}`` (e.g. ``"BOND"``, ``"ENER"``)

    Examples
    --------
    >>> monitors = [
    ...     {"type": "phi", "resid": 2},
    ...     {"type": "energy", "term": "ENER"},
    ...     {"type": "distance", "i": 5, "j": 18},
    ... ]
    >>> collector = Collector(nsteps=10000, nsavc=100, monitors=monitors)
    >>> results = collector.run()
    >>> results["phi(2)"].acf(maxlag=50)
    """

    def __init__(self, nsteps, nsavc, monitors):
        self.nsteps = nsteps
        self.nsavc = nsavc
        self.monitors = monitors
        self._results = {}

    def run(self, dynamics_command="dyna leap verlet start "
            "timestep 0.001 nstep {nsavc} nprint {nsavc} "
            "iprfrq {nsavc} -\niasvel 1 firstt 298.0 finalt 298.0 "
            "iseed 12345"):
        """Execute dynamics and collect monitored quantities.

        Parameters
        ----------
        dynamics_command : str
            CHARMM dynamics command template.  ``{nsavc}`` is replaced
            with the collection frequency.  The command runs *nsavc*
            steps per segment, repeated until *nsteps* total.

        Returns
        -------
        dict of str -> Series
            Mapping from monitor name to collected time series.
        """
        import pycharmm.energy as energy

        # Resolve atom indices for backbone monitors
        resolved = []
        for mon in self.monitors:
            m = dict(mon)
            mtype = m["type"]
            if mtype == "phi":
                resid = m["resid"]
                m["_atoms"] = (
                    _find_atom(resid - 1, "C"),
                    _find_atom(resid, "N"),
                    _find_atom(resid, "CA"),
                    _find_atom(resid, "C"))
                m["_name"] = f"phi({resid})"
            elif mtype == "psi":
                resid = m["resid"]
                m["_atoms"] = (
                    _find_atom(resid, "N"),
                    _find_atom(resid, "CA"),
                    _find_atom(resid, "C"),
                    _find_atom(resid + 1, "N"))
                m["_name"] = f"psi({resid})"
            elif mtype == "distance":
                m["_name"] = f"dist({m['i']},{m['j']})"
            elif mtype == "angle":
                m["_name"] = f"angle({m['i']},{m['j']},{m['k']})"
            elif mtype == "dihedral":
                m["_name"] = f"dihe({m['i']},{m['j']},{m['k']},{m['l']})"
            elif mtype == "energy":
                m["_name"] = m["term"]
            resolved.append(m)

        collectors = {m["_name"]: [] for m in resolved}

        # Set return types for single-frame geometry functions
        lib.correl_distance.restype = ctypes.c_double
        lib.correl_angle.restype = ctypes.c_double
        lib.correl_dihedral.restype = ctypes.c_double

        cmd = dynamics_command.format(nsavc=self.nsavc)
        nsegments = self.nsteps // self.nsavc

        for seg in range(nsegments):
            lingo.charmm_script(cmd)

            # Collect snapshot after this segment
            pos = coor.get_positions()
            x = np.array(pos['x'].tolist(), dtype=np.float64)
            y = np.array(pos['y'].tolist(), dtype=np.float64)
            z = np.array(pos['z'].tolist(), dtype=np.float64)

            for m in resolved:
                mtype = m["type"]
                name = m["_name"]
                if mtype == "distance":
                    i, j = m["i"], m["j"]
                    val = lib.correl_distance(
                        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_int(i), ctypes.c_int(j))
                    collectors[name].append(val)
                elif mtype == "angle":
                    i, j, k = m["i"], m["j"], m["k"]
                    val = lib.correl_angle(
                        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_int(i), ctypes.c_int(j),
                        ctypes.c_int(k))
                    collectors[name].append(val)
                elif mtype in ("dihedral", "phi", "psi"):
                    if "_atoms" in m:
                        atoms = m["_atoms"]
                    else:
                        atoms = (m["i"], m["j"], m["k"], m["l"])
                    i, j, k, l = atoms
                    val = lib.correl_dihedral(
                        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_int(i), ctypes.c_int(j),
                        ctypes.c_int(k), ctypes.c_int(l))
                    collectors[name].append(val)
                elif mtype == "energy":
                    term = m["term"]
                    if term == "ENER":
                        collectors[name].append(
                            energy.get_property_by_name("ENER"))
                    else:
                        try:
                            collectors[name].append(
                                energy.get_term_by_name(term))
                        except ValueError:
                            collectors[name].append(0.0)

        dt = self.nsavc * 0.001  # default timestep assumption
        results = {}
        for m in resolved:
            name = m["_name"]
            units = "kcal/mol" if m["type"] == "energy" else \
                    "Angstrom" if m["type"] == "distance" else "degrees"
            results[name] = Series(collectors[name], name=name,
                                   units=units, dt=dt)

        self._results = results
        return results
