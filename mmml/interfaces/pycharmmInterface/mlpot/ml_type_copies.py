"""Zero CHARMM bonds/angles on ML atoms only, via ML copies of their atom types.

CHARMM bonded parameters are keyed by atom type, so ``READ PARAM APPEND`` of a
zeroed parameter file also zeroes every MM atom of the same type (issue #225).
For hybrid ML/MM registration, each ML atom type ``T`` instead gets a copy
``MLT`` whose bonds and angles have zero force constants (``b0``/``theta0``
kept, so SHAKE and IC builds still work):

1. an appended FLEX parameter file declares the copies (``ATOMS``/``MASS``) and
   lists every bond/angle type tuple of the ML atoms with the copy names;
2. the copies share the original's van der Waals group (``ITC``), so VDW,
   1-4 VDW and NBFIX are unchanged;
3. ``SCALAR TYPE SET`` moves the ML atoms to the copies.

MM atoms keep their types and parameters. Bonds and angles stay in the PSF
(nonbond exclusions, SHAKE and bond-based ML geometry checks keep working);
:func:`restore_ml_atom_types` switches the ML atoms back, e.g. for bonded-MM
recovery. Reference values are read from CHARMM's FLEX parameter tables, not
from ``ICB``/``ICT``, so they do not depend on CHARMM having refreshed its
parameter codes.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import tempfile
from pathlib import Path
from typing import Any

# copy code -> (original code, copy name); codes are 0-based like psf.get_iac().
_copies: dict[int, tuple[int, str]] = {}


def ml_type_copies_active() -> bool:
    return bool(_copies)


def clear_ml_type_copies() -> None:
    """Forget the copies (the PSF was reloaded with the original types)."""
    _copies.clear()


def copy_type_name(orig: str, taken: set[str]) -> str:
    """Name of the ML copy of atom type *orig* (at most 8 characters, unique)."""
    name = f"ML{orig}"
    if len(name) <= 8 and name not in taken:
        return name
    for k in range(100000):
        name = f"MLX{k:05d}"
        if name not in taken:
            return name
    raise RuntimeError("no free ML atom type name")


def _fortran_symbol(lib: Any, var: str, ctype: Any) -> Any:
    """Module variable ``var`` of CHARMM (gfortran or Intel name mangling)."""
    module, name = var.split(".")
    for symbol in (f"__{module}_MOD_{name}", f"{module}_mp_{name}_"):
        try:
            return ctype.in_dll(lib, symbol)
        except ValueError:
            continue
    raise RuntimeError(f"CHARMM symbol {var} not exported by libcharmm")


def _static_array(lib: Any, var: str, ctype: Any, n: int) -> Any:
    return (ctype * n).from_address(ctypes.addressof(_fortran_symbol(lib, var, ctype)))


def _allocatable_array(lib: Any, var: str, ctype: Any, n: int) -> Any:
    """First *n* elements of an allocatable array (descriptor starts with base_addr)."""
    base = _fortran_symbol(lib, var, ctypes.c_void_p).value
    if not base:
        raise RuntimeError(f"CHARMM array {var} is not allocated")
    return (ctype * n).from_address(base)


def flex_lookup(key: tuple[int, ...], columns: list[Any], n: int) -> int | None:
    """CHARMM FLEX parameter search (CODES): last matching entry, either direction.

    *key* and the *columns* hold 1-based atom type codes. Negative table codes
    are FLEX equivalence groups; they are not resolved here, so meeting one
    before a match raises.
    """
    rev = tuple(reversed(key))
    for j in range(n - 1, -1, -1):
        entry = tuple(int(col[j]) for col in columns)
        if entry == key or entry == rev:
            return j
        if min(entry) < 0:
            raise RuntimeError(
                "ML type copies: FLEX equivalence parameters are not supported "
                "(use --mlpot-use-block)"
            )
    return None


def ml_bonded_copy_prm_text(
    masses: dict[str, float],
    bonds: dict[tuple[str, str], float],
    angles: dict[tuple[str, str, str], float],
) -> str:
    """FLEX parameter file: copy types plus their zero-force bonds/angles."""
    lines = ["* mmml: ML copies of CGenFF atom types (zero bonded force constants)", "*", ""]
    if masses:
        lines.append("ATOMS")
        lines += [f"MASS -1 {name:<8s} {mass:10.5f}" for name, mass in masses.items()]
        lines.append("")
    lines.append("BONDS")
    lines += [f"{a:<8s} {b:<8s} 0.0 {b0:10.4f}" for (a, b), b0 in bonds.items()]
    lines += ["", "ANGLES"]
    lines += [
        f"{a:<8s} {b:<8s} {c:<8s} 0.0 {theta0:10.4f}" for (a, b, c), theta0 in angles.items()
    ]
    lines += ["", "END", ""]
    return "\n".join(lines)


def _ml_bonded_terms(pycharmm: Any, ml: set[int]) -> dict[str, list[tuple[int, ...]]]:
    """0-based atom tuples of the PSF bonds and angles that touch an ML atom."""
    lib = pycharmm.lib.charmm
    terms: dict[str, list[tuple[int, ...]]] = {"bonds": [], "angles": []}
    if int(pycharmm.psf.get_nbond()):
        ib, jb = pycharmm.psf.get_ib_jb()
        terms["bonds"] = [(int(i) - 1, int(j) - 1) for i, j in zip(ib, jb, strict=True)]
    ntheta = int(_fortran_symbol(lib, "psf.ntheta", ctypes.c_int).value)
    if ntheta:
        cols = [_allocatable_array(lib, f"psf.{v}", ctypes.c_int, ntheta) for v in ("it", "jt", "kt")]
        terms["angles"] = [tuple(int(c[k]) - 1 for c in cols) for k in range(ntheta)]
    for kind, rows in terms.items():
        terms[kind] = [t for t in rows if min(t) >= 0 and ml.intersection(t)]
    return terms


# FLEX parameter tables: (count, type-code columns, reference value, to degrees).
_TABLES = {
    "bonds": ("param.ncb", ("param.cbai", "param.cbaj"), "param.cbb", False),
    "angles": ("param.nct", ("param.ctai", "param.ctaj", "param.ctak"), "param.ctb", True),
}


def apply_ml_type_copies(
    ml_indices: Any, ml_store_name: str, *, pycharmm: Any = None
) -> dict[str, int]:
    """Move ML atoms to zero-bonded copies of their types (see module docstring).

    *ml_store_name* is the stored CHARMM selection of the ML atoms. Returns the
    number of copied types and of bond/angle parameter rows written.
    """
    if pycharmm is None:
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import _import_pycharmm

        pycharmm = _import_pycharmm()
    param = getattr(pycharmm, "param", None) or importlib.import_module("pycharmm.param")
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    restore_ml_atom_types(pycharmm=pycharmm)
    lib = pycharmm.lib.charmm
    ml = {int(i) for i in ml_indices}
    iac = [int(c) for c in pycharmm.psf.get_iac()]
    atc = [str(t).strip() for t in param.get_atc()]
    amass = list(pycharmm.psf.get_amass())

    taken = set(atc)
    new_name: dict[int, str] = {}
    masses: dict[str, float] = {}
    for i in sorted(ml):
        code = iac[i]
        if code not in new_name:
            name = f"ML{atc[code]}"
            if name not in taken:  # else: copy left by an earlier registration
                name = copy_type_name(atc[code], taken)
                taken.add(name)
                masses[name] = float(amass[i])
            new_name[code] = name

    # One zero-force row per type pattern, with copy names only on ML atoms.
    rows: dict[str, dict[tuple[str, ...], float]] = {}
    for kind, atoms_list in _ml_bonded_terms(pycharmm, ml).items():
        count, cols, ref, degrees = _TABLES[kind]
        n = int(_fortran_symbol(lib, count, ctypes.c_int).value)
        columns = [_static_array(lib, c, ctypes.c_int, n) for c in cols]
        values = _static_array(lib, ref, ctypes.c_double, n)
        rows[kind] = {}
        for atoms in atoms_list:
            codes = tuple(iac[a] for a in atoms)
            hit = flex_lookup(tuple(c + 1 for c in codes), columns, n)
            if hit is None:
                raise RuntimeError(
                    f"ML type copies: no {kind} parameters for "
                    f"{'-'.join(atc[c] for c in codes)}"
                )
            value = math.degrees(values[hit]) if degrees else float(values[hit])
            key = tuple(new_name[c] if a in ml else atc[c] for c, a in zip(codes, atoms, strict=True))
            rows[kind][key] = value

    text = ml_bonded_copy_prm_text(masses, rows["bonds"], rows["angles"])  # type: ignore[arg-type]
    with tempfile.TemporaryDirectory(prefix="mmml_mltypes_") as tmp:
        path = Path(tmp) / "ml_type_copies.prm"
        path.write_text(text)
        read_cgenff_prm(path, append=True)

    atc_after = [str(t).strip() for t in param.get_atc()]
    itc = _static_array(lib, "param.itc", ctypes.c_int, len(atc_after))
    vdw_before = (list(param.get_vdwr()), list(param.get_epsilon()))
    for code, name in new_name.items():
        copy = atc_after.index(name)
        # Same VDW group as the original: VDW, 1-4 VDW and NBFIX unchanged.
        itc[copy] = itc[code]
        _copies[copy] = (code, name)
        pycharmm.lingo.charmm_script(
            f"SCALAR TYPE SET {copy + 1} SELE {ml_store_name.upper()} .AND. CHEM {atc[code]} END"
        )
    moved = sorted(i for i, c in enumerate(pycharmm.psf.get_iac()) if int(c) in _copies)
    vdw_after = (list(param.get_vdwr()), list(param.get_epsilon()))
    if moved != sorted(ml) or vdw_after != vdw_before:
        raise RuntimeError(
            "ML type copies: atom types or VDW parameters differ after the switch "
            f"({len(moved)} of {len(ml)} ML atoms moved)"
        )
    return {"types": len(new_name), "bonds": len(rows["bonds"]), "angles": len(rows["angles"])}


def restore_ml_atom_types(*, pycharmm: Any = None) -> bool:
    """Move ML atoms back to their original types; True if anything was restored."""
    if not _copies:
        return False
    if pycharmm is None:
        from mmml.interfaces.pycharmmInterface.mlpot.block_terms import _import_pycharmm

        pycharmm = _import_pycharmm()
    for code, name in _copies.values():
        pycharmm.lingo.charmm_script(f"SCALAR TYPE SET {code + 1} SELE CHEM {name} END")
    _copies.clear()
    return True
