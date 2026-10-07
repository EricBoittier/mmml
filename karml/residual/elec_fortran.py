"""Python front-ends for the Fortran electrostatics kernel.

Two paths:
  1. f2py module `elec_f2py` (import directly, easiest, handles arrays for you)
  2. ctypes against `libelec.so` (no numpy.f2py runtime dep; explicit pointers)

Both call the SAME Fortran math as electrostatics_jax.py, so use whichever is
convenient — the residual targets are identical to float precision.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

KE_KCAL_ANG = 332.0637128
_HERE = os.path.dirname(__file__)


# --------------------------------------------------------------------------
# Path 1: f2py module
# --------------------------------------------------------------------------
def ef_f2py(R, q, pi, pj, ke=KE_KCAL_ANG, r_on=10.0, r_off=12.0, use_switch=False):
    """Requires: f2py -c -m elec_f2py elec_kernel.f90 --f90flags='-fopenmp' -lgomp"""
    import elec_f2py  # built extension module

    # Fortran wants column-major (3, n). asfortranarray avoids silent copies.
    Rf = np.asfortranarray(np.asarray(R, dtype=np.float64).T)   # (3, n)
    q = np.ascontiguousarray(q, dtype=np.float64)
    pi1 = np.ascontiguousarray(pi, dtype=np.int32) + 1          # 1-based
    pj1 = np.ascontiguousarray(pj, dtype=np.int32) + 1
    energy, F = elec_f2py.elec_ef(
        Rf, q, pi1, pj1, ke, r_on, r_off, int(use_switch)
    )
    return float(energy), np.asarray(F).T.copy()               # back to (n, 3)


# --------------------------------------------------------------------------
# Path 2: ctypes against libelec.so
# --------------------------------------------------------------------------
_lib = None


def _load(libpath=None):
    global _lib
    if _lib is not None:
        return _lib
    libpath = libpath or os.path.join(_HERE, "libelec.so")
    lib = ctypes.CDLL(libpath)
    c_dbl_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.elec_ef_c.restype = None
    lib.elec_ef_c.argtypes = [
        ctypes.c_int, ctypes.c_int,        # n, p (by value)
        c_dbl_p, c_dbl_p,                  # R(3,n), q(n)
        c_int_p, c_int_p,                  # pi(p), pj(p)
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # ke, r_on, r_off
        ctypes.c_int,                      # use_switch (by value)
        c_dbl_p, c_dbl_p,                  # energy (scalar as (1,)), F(3,n)
    ]
    _lib = lib
    return lib


def ef_ctypes(R, q, pi, pj, ke=KE_KCAL_ANG, r_on=10.0, r_off=12.0,
              use_switch=False, libpath=None):
    """Requires: gfortran -O3 -fopenmp -shared -fPIC -o libelec.so elec_kernel.f90"""
    lib = _load(libpath)
    R = np.asarray(R, dtype=np.float64)
    n = R.shape[0]
    Rf = np.asfortranarray(R.T)                       # (3, n) col-major
    q = np.ascontiguousarray(q, dtype=np.float64)
    pi1 = np.ascontiguousarray(pi, dtype=np.int32) + 1
    pj1 = np.ascontiguousarray(pj, dtype=np.int32) + 1
    p = pi1.shape[0]
    energy = np.zeros(1, dtype=np.float64)
    F = np.asfortranarray(np.zeros((3, n), dtype=np.float64))
    lib.elec_ef_c(
        n, p, Rf, q, pi1, pj1,
        float(ke), float(r_on), float(r_off), int(use_switch),
        energy, F,
    )
    return float(energy[0]), np.ascontiguousarray(F.T)          # (n, 3)
