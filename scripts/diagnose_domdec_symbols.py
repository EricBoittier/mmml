#!/usr/bin/env python3
"""Diagnose CHARMM DOMDEC symbol names in libcharmm.so.

Usage (on the cluster, with the mmml venv active):
    python scripts/diagnose_domdec_symbols.py

What it does:
  1. Finds libcharmm.so via PyCHARMM or CHARMM_LIB_DIR.
  2. Runs 'nm -D' on the library and lists all domdec-related symbols.
  3. Probes likely variable names for NDIR, natom counts, and atom index arrays.
  4. Prints the correct symbol names to paste back into domdec_atoms.py.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Find libcharmm.so
# ---------------------------------------------------------------------------

def find_libcharmm() -> str | None:
    # Try via PyCHARMM
    try:
        import pycharmm.lib as lib  # noqa: PLC0415
        handle = lib.charmm._handle
        # On Linux, /proc/self/maps lets us map the handle back to a path
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    if "libcharmm" in line:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            return parts[-1]
        except OSError:
            pass
        # Fallback: ask ldconfig / ldd
        for candidate in [
            os.environ.get("CHARMM_LIB_DIR", ""),
        ]:
            path = os.path.join(candidate, "libcharmm.so")
            if os.path.exists(path):
                return path
    except ImportError:
        pass

    # Try env
    lib_dir = os.environ.get("CHARMM_LIB_DIR", "")
    if lib_dir:
        p = os.path.join(lib_dir, "libcharmm.so")
        if os.path.exists(p):
            return p

    # Search PATH-adjacent lib directories
    for d in os.environ.get("PATH", "").split(":"):
        for suffix in [
            "../lib/libcharmm.so",
            "../../lib/libcharmm.so",
            "../lib64/libcharmm.so",
        ]:
            p = os.path.normpath(os.path.join(d, suffix))
            if os.path.exists(p):
                return p

    return None


# ---------------------------------------------------------------------------
# nm symbol scan
# ---------------------------------------------------------------------------

def nm_domdec_symbols(lib_path: str) -> list[str]:
    """Return all exported symbols that contain 'domdec' (case-insensitive)."""
    try:
        out = subprocess.check_output(
            ["nm", "-D", lib_path],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nm not available or not a shared lib — fall back to strings grep
        try:
            out = subprocess.check_output(
                ["strings", lib_path],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return []
    lines = [l for l in out.splitlines() if "domdec" in l.lower()]
    # Extract the symbol name (last field for nm output, full line for strings)
    symbols = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            sym = parts[-1]
            if sym.startswith("__") and "_MOD_" in sym:
                symbols.append(sym)
    return sorted(set(symbols))


# ---------------------------------------------------------------------------
# Probe known candidate names for NDIR / atom-count / array variables
# ---------------------------------------------------------------------------

def _try_int32(lib: ctypes.CDLL, sym: str) -> int | None:
    try:
        return ctypes.c_int32.in_dll(lib, sym).value
    except (OSError, ValueError, AttributeError, TypeError):
        return None


NDIR_CANDIDATES = [
    # (x, y, z) triples to try
    ("ndomx",   "ndomy",   "ndomz"),
    ("ndirx",   "ndiry",   "ndirz"),
    ("ndir_x",  "ndir_y",  "ndir_z"),
    ("nnx",     "nny",     "nnz"),
    ("ndom_x",  "ndom_y",  "ndom_z"),
]

NATOM_LOCAL_CANDIDATES   = ["natoml", "nl_local", "natom_local", "nlocal"]
NATOM_FOREIGN_CANDIDATES = ["natom_foreign", "nf_foreign", "nforeign", "nghost", "natom_ghost"]
IIML_CANDIDATES          = ["iiml", "ilocal", "il_local", "local_atoms"]
IIMF_CANDIDATES          = ["iimf", "iforeign", "if_foreign", "ghost_atoms"]

_MODULE_COMMON = "domdec_common"
_MODULE_DR     = "domdec_dr_common"
_MODULES       = [_MODULE_COMMON, _MODULE_DR, "domdec"]


def _sym(module: str, var: str) -> str:
    return f"__{module.lower()}_MOD_{var.lower()}"


def probe_ndir(lib: ctypes.CDLL) -> tuple[str, str, str] | None:
    """Return the (sym_x, sym_y, sym_z) triple that has non-None values, or None."""
    for mod in _MODULES:
        for nx_name, ny_name, nz_name in NDIR_CANDIDATES:
            sx, sy, sz = _sym(mod, nx_name), _sym(mod, ny_name), _sym(mod, nz_name)
            vx = _try_int32(lib, sx)
            vy = _try_int32(lib, sy)
            vz = _try_int32(lib, sz)
            if vx is not None and vy is not None and vz is not None:
                return (sx, sy, sz), (vx, vy, vz)
    return None, None


def probe_scalar(lib: ctypes.CDLL, candidates: list[str]) -> tuple[str, int] | tuple[None, None]:
    for mod in _MODULES:
        for var in candidates:
            sym = _sym(mod, var)
            val = _try_int32(lib, sym)
            if val is not None:
                return sym, val
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CHARMM DOMDEC symbol probe")
    print("=" * 60)

    # --- find library ---
    lib_path = find_libcharmm()
    if lib_path is None:
        print("ERROR: libcharmm.so not found.")
        print("  Set CHARMM_LIB_DIR=<dir> or import pycharmm before running.")
        sys.exit(1)
    print(f"Library : {lib_path}\n")

    # --- nm scan ---
    all_domdec_syms = nm_domdec_symbols(lib_path)
    print(f"All domdec symbols in library ({len(all_domdec_syms)}):")
    for s in all_domdec_syms:
        print(f"  {s}")
    print()

    # --- load and probe ---
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"ERROR loading library: {e}")
        sys.exit(1)

    # q_domdec
    q_sym = _sym(_MODULE_COMMON, "q_domdec")
    q_val = _try_int32(lib, q_sym)
    print(f"q_domdec ({q_sym}): {q_val!r}  {'✓' if q_val is not None else '✗ NOT FOUND'}")

    # natoml
    natoml_sym, natoml_val = probe_scalar(lib, NATOM_LOCAL_CANDIDATES)
    print(f"natoml   ({natoml_sym}): {natoml_val!r}  {'✓' if natoml_sym else '✗ NOT FOUND'}")

    # natom_foreign
    nf_sym, nf_val = probe_scalar(lib, NATOM_FOREIGN_CANDIDATES)
    print(f"natom_foreign ({nf_sym}): {nf_val!r}  {'✓' if nf_sym else '✗ NOT FOUND'}")

    # NDIR
    ndir_syms, ndir_vals = probe_ndir(lib)
    if ndir_syms:
        sx, sy, sz = ndir_syms
        vx, vy, vz = ndir_vals
        print(f"NDIR x   ({sx}): {vx}  ✓")
        print(f"NDIR y   ({sy}): {vy}  ✓")
        print(f"NDIR z   ({sz}): {vz}  ✓")
    else:
        print("NDIR     : ✗ NOT FOUND — try scanning all_domdec_syms above")

    # iiml / iimf
    iiml_sym, _ = probe_scalar(lib, IIML_CANDIDATES)
    iimf_sym, _ = probe_scalar(lib, IIMF_CANDIDATES)
    print(f"iiml     ({iiml_sym}): {'✓' if iiml_sym else '✗ NOT FOUND'}")
    print(f"iimf     ({iimf_sym}): {'✓' if iimf_sym else '✗ NOT FOUND'}")

    # --- summary ---
    print()
    print("=" * 60)
    print("Paste the following into domdec_atoms.py to fix symbol names:")
    print("=" * 60)
    if ndir_syms:
        sx, sy, sz = ndir_syms
        # Extract the var name from the full symbol
        vx_name = sx.split("_MOD_")[1]
        vy_name = sy.split("_MOD_")[1]
        vz_name = sz.split("_MOD_")[1]
        mod_name = sx.split("_MOD_")[0].lstrip("_")
        print(f'_MODULE = "{mod_name}"')
        print(f'_SYM_NDOMX = _sym("{vx_name}")')
        print(f'_SYM_NDOMY = _sym("{vy_name}")')
        print(f'_SYM_NDOMZ = _sym("{vz_name}")')
    if nf_sym:
        nf_var = nf_sym.split("_MOD_")[1]
        print(f'_SYM_NATOM_FOREIGN = _sym("{nf_var}")')
    if iiml_sym:
        iiml_var = iiml_sym.split("_MOD_")[1]
        print(f'_SYM_IIML = _sym("{iiml_var}")')
    if iimf_sym:
        iimf_var = iimf_sym.split("_MOD_")[1]
        print(f'_SYM_IIMF = _sym("{iimf_var}")')

    if not ndir_syms:
        print()
        print("NDIR symbols not found by brute-force probe.")
        print("Look for 'ndir' or 'ndom' in the nm output above and")
        print("add the correct names to NDIR_CANDIDATES in this script.")


if __name__ == "__main__":
    # Bootstrap: import pycharmm first so libcharmm.so is loaded into the process
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    except Exception:
        pass
    main()
