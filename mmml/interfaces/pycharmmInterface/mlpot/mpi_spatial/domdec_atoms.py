"""Read CHARMM DOMDEC atom maps from libcharmm.so via ctypes.

Architecture
------------
CHARMM stores domain-decomposition atom ownership in the Fortran module
``domdec_common`` (source: ``source/domdec/domdec_dr_common.F90``).

With gfortran, module variables are exported as::

    __<module>_MOD_<variable>

Available data (read without upstream PyCHARMM changes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scalars — ``ctypes.c_int32.in_dll`` or ``c_int8.in_dll``:

  q_domdec      – LOGICAL(4): ``.TRUE.`` when DOMDEC is currently active
  ndomx/y/z     – INTEGER(4): NDIR axis counts (domains per axis)
  natoml        – INTEGER(4): number of local atoms owned by this rank
  natom_foreign – INTEGER(4): number of ghost atoms visible to this rank

Arrays — 1-D ``ALLOCATABLE INTEGER(4)`` read via gfortran internal descriptor:

  iiml  – shape ``(natoml)``: 1-based global atom indices owned by this rank
  iimf  – shape ``(natom_foreign)``: 1-based global ghost atom indices

All Python-facing functions return **0-based** ``np.int32`` arrays.

Fallback
--------
Every function returns a safe default (``False``, ``(1,1,1)``, empty array)
when the symbol is absent.  This covers: DOMDEC compiled out (``KEY_DOMDEC=0``),
non-gfortran compilers, or PyCHARMM not yet imported (``libcharmm.so`` not
loaded into the process).

Descriptor layout (gfortran 64-bit, all 8-byte fields)
-------------------------------------------------------
::

    struct {
        void     *base_addr;   // NULL when unallocated
        intptr_t  offset;      // element offset (usually 1 for 1-based indexing)
        int64_t   dtype;       // packed: bits[2:0]=type, bits[5:3]=kind, bits[8:6]=rank
        int64_t   span;        // element size in bytes (4 for INTEGER(4))
        // per-dimension triple — STRIDE first, then lbound, then ubound:
        int64_t   dim0_stride; // stride in elements (1 for contiguous)
        int64_t   dim0_lbound; // lower bound (1 for standard Fortran arrays)
        int64_t   dim0_ubound; // upper bound (= natoml for iiml)
    };

References
----------
- gfortran internals: libgfortran/libgfortran.h  (``array_t``, ``descriptor_dimension``)
- CHARMM c47 domdec module: ``source/domdec/domdec_dr_common.F90``
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-variable symbol names (gfortran name mangling)
#
# Confirmed by nm -D on libcharmm.so (c47, June 2026):
#   domdec_common  : q_domdec, nx, ny, nz, natoml, natoml_tot, atoml,
#                    boxx/y/z, homeix/y/z, ndirect, nneigh, ...
#   domdec_local   : loc2glo_ind (local→global atom index array),
#                    glo2loc_ind, xyzq_loc, sforce, ...
#   domdec_dr_common: ndir_set, q_recip_node, q_direct_node, ...
# ---------------------------------------------------------------------------

def _sym(module: str, variable: str) -> str:
    """Return the gfortran-mangled symbol for a Fortran module variable."""
    return f"__{module.lower()}_MOD_{variable.lower()}"


_MOD_COMMON = "domdec_common"
_MOD_LOCAL  = "domdec_local"

# Scalars — domdec_common
_SYM_Q_DOMDEC      = _sym(_MOD_COMMON, "q_domdec")
_SYM_NDOMX         = _sym(_MOD_COMMON, "nx")        # confirmed: 'nx' not 'ndomx'
_SYM_NDOMY         = _sym(_MOD_COMMON, "ny")
_SYM_NDOMZ         = _sym(_MOD_COMMON, "nz")
_SYM_NATOML        = _sym(_MOD_COMMON, "natoml")    # number of local atoms
_SYM_NATOM_FOREIGN = _sym(_MOD_COMMON, "natoml_tot") # total atoms incl. ghost zones

# 1-D allocatable INTEGER arrays
# domdec_local::loc2glo_ind — local index → global 1-based atom index
_SYM_IIML = _sym(_MOD_LOCAL, "loc2glo_ind")
# No direct ghost-index array found; ghost atoms are zones 1..nneigh of atoml.
# Use natoml_tot - natoml to derive ghost count; indices via atoml[natoml:natoml_tot].
_SYM_ATOML = _sym(_MOD_COMMON, "atoml")   # full atom list (local + ghost zones)


# ---------------------------------------------------------------------------
# gfortran 1-D array descriptor (64-bit, contiguous allocatable)
# ---------------------------------------------------------------------------

class _GF1DArrayDescriptor(ctypes.Structure):
    """Internal gfortran array descriptor for a 1-D allocatable array (64-bit).

    Field order matches ``libgfortran/libgfortran.h``: stride → lbound → ubound
    for each dimension.
    """

    _fields_ = [
        ("base_addr", ctypes.c_void_p),   # NULL when unallocated
        ("offset", ctypes.c_int64),        # element offset (1-based → offset=1)
        ("dtype", ctypes.c_int64),         # packed type+kind+rank
        ("span", ctypes.c_int64),          # element size in bytes
        ("dim0_stride", ctypes.c_int64),   # stride (elements)
        ("dim0_lbound", ctypes.c_int64),   # lower bound (usually 1)
        ("dim0_ubound", ctypes.c_int64),   # upper bound (= length for lb=1)
    ]


# ---------------------------------------------------------------------------
# Library handle
# ---------------------------------------------------------------------------

def _get_libcharmm() -> Optional[ctypes.CDLL]:
    """Return the already-loaded ``libcharmm.so`` handle, or ``None``.

    Prefers the handle that PyCHARMM already loaded (avoids a second ``dlopen``
    which would give a separate address space).  Falls back to ``ctypes.CDLL``
    with the path from ``CHARMM_LIB_DIR``.
    """
    # 1. Use PyCHARMM's own handle — most reliable because it's the same mmap
    try:
        import pycharmm.lib as lib  # noqa: PLC0415

        return lib.charmm
    except Exception:
        pass

    # 2. Try env-driven path
    lib_dir = os.environ.get("CHARMM_LIB_DIR", "")
    candidates = [
        os.path.join(lib_dir, "libcharmm.so"),
        "libcharmm.so",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass

    # 3. ctypes.util search (works if LD_LIBRARY_PATH is set correctly)
    found = ctypes.util.find_library("charmm")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            pass

    return None


# ---------------------------------------------------------------------------
# Scalar readers
# ---------------------------------------------------------------------------

def _read_int32(sym: str) -> Optional[int]:
    """Read a Fortran INTEGER(4) module variable.  Returns ``None`` on failure."""
    lib = _get_libcharmm()
    if lib is None:
        return None
    try:
        return int(ctypes.c_int32.in_dll(lib, sym).value)
    except (OSError, ValueError, AttributeError):
        return None


def _read_logical4(sym: str) -> Optional[bool]:
    """Read a Fortran LOGICAL(4) module variable.  Returns ``None`` on failure.

    Fortran LOGICAL is typically 4 bytes; any non-zero value is ``.TRUE.``.
    """
    lib = _get_libcharmm()
    if lib is None:
        return None
    try:
        return bool(ctypes.c_int32.in_dll(lib, sym).value != 0)
    except (OSError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Array reader (via gfortran internal descriptor)
# ---------------------------------------------------------------------------

def _read_int32_1d_array(sym: str) -> Optional[np.ndarray]:
    """Read a 1-D ``ALLOCATABLE INTEGER(4)`` Fortran array.

    Returns a **copy** as ``np.int32`` (1-based raw values, no index shift).
    Returns ``None`` on failure or when the array is unallocated.
    """
    lib = _get_libcharmm()
    if lib is None:
        return None
    try:
        desc = _GF1DArrayDescriptor.in_dll(lib, sym)
    except (OSError, ValueError, AttributeError, TypeError):
        return None

    if not desc.base_addr:
        log.debug("domdec_atoms: %s is unallocated (NULL base_addr)", sym)
        return None

    n = int(desc.dim0_ubound - desc.dim0_lbound + 1)
    if n <= 0:
        return np.empty(0, dtype=np.int32)

    arr_t = ctypes.c_int32 * n
    try:
        arr = arr_t.from_address(desc.base_addr)
    except (ValueError, OSError):
        log.warning("domdec_atoms: could not read %s data (n=%d)", sym, n)
        return None

    return np.frombuffer(arr, dtype=np.int32).copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_domdec_active() -> bool:
    """``True`` when CHARMM DOMDEC is currently active (``q_domdec == .TRUE.``).

    Returns ``False`` when the symbol is absent (DOMDEC not compiled in).
    """
    val = _read_logical4(_SYM_Q_DOMDEC)
    return bool(val) if val is not None else False


def get_ndir() -> tuple[int, int, int]:
    """CHARMM DOMDEC NDIR axes ``(Nx, Ny, Nz)``.

    Returns ``(1, 1, 1)`` when DOMDEC is inactive or symbols are absent.
    """
    nx = _read_int32(_SYM_NDOMX)
    ny = _read_int32(_SYM_NDOMY)
    nz = _read_int32(_SYM_NDOMZ)
    return (
        max(1, int(nx)) if nx is not None else 1,
        max(1, int(ny)) if ny is not None else 1,
        max(1, int(nz)) if nz is not None else 1,
    )


def get_local_atom_count() -> int:
    """Number of atoms owned by this MPI rank under DOMDEC (``natoml``).

    Returns ``0`` when DOMDEC is inactive or symbol absent.
    """
    val = _read_int32(_SYM_NATOML)
    return max(0, int(val)) if val is not None else 0


def get_ghost_atom_count() -> int:
    """Number of ghost atoms visible to this rank under DOMDEC.

    Computed as ``natoml_tot - natoml`` (total atoms in all zones minus local).
    Returns ``0`` when DOMDEC is inactive or symbols absent.
    """
    tot = _read_int32(_SYM_NATOM_FOREIGN)   # natoml_tot
    loc = _read_int32(_SYM_NATOML)
    if tot is None or loc is None:
        return 0
    return max(0, int(tot) - int(loc))


def get_local_atom_indices() -> np.ndarray:
    """0-based global atom indices owned by this MPI rank under DOMDEC.

    Reads ``domdec_local::loc2glo_ind`` (local-index → global 1-based).
    Only the first ``natoml`` entries are local; the array may be larger.

    Returns an empty ``np.int32`` array when DOMDEC is inactive, the symbol is
    absent, or the array is not yet allocated (called before dynamics starts).
    """
    nlocal = get_local_atom_count()
    if nlocal == 0:
        return np.empty(0, dtype=np.int32)
    raw = _read_int32_1d_array(_SYM_IIML)
    if raw is None or raw.size == 0:
        return np.empty(0, dtype=np.int32)
    # Trim to nlocal entries (array may be allocated larger) and convert to 0-based
    return (raw[:nlocal] - 1).astype(np.int32)


def get_ghost_atom_indices() -> np.ndarray:
    """0-based global atom indices of ghost atoms visible to this rank.

    Reads ``domdec_common::atoml`` (full zone list) and returns entries
    from ``natoml`` to ``natoml_tot - 1`` (the imported ghost zones).

    Returns an empty ``np.int32`` array when DOMDEC is inactive or absent.
    """
    nlocal = get_local_atom_count()
    ntot_val = _read_int32(_SYM_NATOM_FOREIGN)   # natoml_tot
    if ntot_val is None:
        return np.empty(0, dtype=np.int32)
    ntot = int(ntot_val)
    nghost = ntot - nlocal
    if nghost <= 0:
        return np.empty(0, dtype=np.int32)
    raw = _read_int32_1d_array(_SYM_ATOML)
    if raw is None or raw.size < ntot:
        return np.empty(0, dtype=np.int32)
    return (raw[nlocal:ntot] - 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def discover_domdec_symbols() -> dict[str, bool]:
    """Probe which ``domdec_common`` symbols are present in ``libcharmm.so``.

    Useful for diagnosing a DOMDEC build on a new cluster::

        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            discover_domdec_symbols,
        )
        print(discover_domdec_symbols())

    Returns a dict mapping symbol name → ``True/False``.
    """
    lib = _get_libcharmm()
    if lib is None:
        return {"_libcharmm_not_found": False}

    symbols = [
        _SYM_Q_DOMDEC,
        _SYM_NDOMX,
        _SYM_NDOMY,
        _SYM_NDOMZ,
        _SYM_NATOML,
        _SYM_NATOM_FOREIGN,
        _SYM_IIML,
        _SYM_ATOML,
    ]
    result: dict[str, bool] = {}
    for sym in symbols:
        try:
            ctypes.c_int32.in_dll(lib, sym)
            result[sym] = True
        except (OSError, ValueError):
            result[sym] = False
    return result


def domdec_summary() -> str:
    """Human-readable summary of the current DOMDEC state."""
    active = is_domdec_active()
    nx, ny, nz = get_ndir()
    nlocal = get_local_atom_count()
    nghost = get_ghost_atom_count()
    sym_avail = discover_domdec_symbols()
    # Filter out the special sentinel key (_libcharmm_not_found)
    real_syms = {k: v for k, v in sym_avail.items() if not k.startswith("_lib")}
    n_found = sum(real_syms.values())
    n_total = len(real_syms)
    lines = [
        f"DOMDEC active    : {active}",
        f"NDIR             : {nx} {ny} {nz}",
        f"Local atoms      : {nlocal}",
        f"Ghost atoms      : {nghost}",
        f"Symbols found    : {n_found}/{n_total}",
    ]
    if n_found < n_total:
        missing = [k for k, v in sym_avail.items() if not v and not k.startswith("_")]
        lines.append(f"Missing symbols  : {', '.join(missing)}")
    return "\n".join(lines)
