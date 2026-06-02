"""CHARMM periodic box setup for MLpot workflows (matches ``md_pbc_suite/ase.py``)."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np


def _charmm_ctypes_scalar(value: Any) -> float:
    """Normalize CHARMM ctypes outputs (may be bare int/float or ctypes wrappers)."""
    if hasattr(value, "value"):
        return float(value.value)
    return float(value)


def cubic_box_length_from_geometry(
    positions: np.ndarray,
    *,
    ml_cutoff: float = 12.0,
    pad: float = 10.0,
) -> float:
    """Auto cubic box side (Å) from cluster span + padding."""
    r = np.asarray(positions, dtype=float)
    span = float(np.max(r.max(axis=0) - r.min(axis=0)))
    return max(span + 2.0 * float(pad), 2.0 * float(ml_cutoff) + float(pad))


def _read_charmm_box_sides_A() -> tuple[float, float, float]:
    """Read CHARMM periodic box lengths (Å) via ``pbound_get_size``."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm.lib as lib

    size_x = ctypes.c_double(0.0)
    size_y = ctypes.c_double(0.0)
    size_z = ctypes.c_double(0.0)
    lib.charmm.pbound_get_size(
        ctypes.byref(size_x),
        ctypes.byref(size_y),
        ctypes.byref(size_z),
    )
    return (
        _charmm_ctypes_scalar(size_x),
        _charmm_ctypes_scalar(size_y),
        _charmm_ctypes_scalar(size_z),
    )


def _is_cubic_box_sides(
    lx: float,
    ly: float,
    lz: float,
    *,
    rel_tol: float = 1e-3,
) -> bool:
    if min(lx, ly, lz) <= 0.0:
        return False
    mean = (lx + ly + lz) / 3.0
    return max(abs(lx - mean), abs(ly - mean), abs(lz - mean)) <= rel_tol * mean


def resolve_charmm_cubic_box_side_A(
    *,
    fallback_side_A: float | None = None,
    rel_tol: float = 1e-3,
) -> tuple[float, bool]:
    """Return ``(side, used_fallback)`` for the current CHARMM cubic cell.

    Uses box side lengths rather than ``pbound_is_cubic_box()`` (that flag can be
    unset between dynamics stages even for a cubic cell).
    """
    lx, ly, lz = _read_charmm_box_sides_A()
    if _is_cubic_box_sides(lx, ly, lz, rel_tol=rel_tol):
        return (lx + ly + lz) / 3.0, False
    if fallback_side_A is not None and float(fallback_side_A) > 0.0:
        return float(fallback_side_A), True
    if min(lx, ly, lz) > 0.0:
        raise RuntimeError(
            "CHARMM box is not cubic; MLpot MIC sync expects a cubic cell "
            f"(got {lx:.4f}, {ly:.4f}, {lz:.4f} Å)"
        )
    raise RuntimeError(
        "CHARMM cubic box side unavailable (pbound_get_size returned non-positive lengths)"
    )


def get_charmm_cubic_box_side_A() -> float:
    """Read the current cubic CHARMM cell side length (Å) from ``pbound_get_size``."""
    side, _ = resolve_charmm_cubic_box_side_A()
    return side


def cubic_box_matrix_from_side(side_A: float) -> np.ndarray:
    """Return orthorhombic 3×3 cell matrix for a cubic box side (Å)."""
    L = float(side_A)
    return np.array([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]], dtype=np.float64)


def prepare_charmm_pbc(cubic_box_side_A: float) -> None:
    """Install CHARMM crystal + IMAGE for a cubic cell."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset
    from mmml.interfaces.pycharmmInterface.setupBox import _ensure_crystal_image_str

    L = float(cubic_box_side_A)
    if L <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {L}")
    _ensure_crystal_image_str()
    pyci.pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=L))
    pyci.pycharmm.lingo.charmm_script(
        "open read unit 10 card name crystal_image.str\n"
        f"crystal defi cubic {L} {L} {L} 90. 90. 90.\n"
        "CRYSTAL READ UNIT 10 CARD\n"
        "image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end\n"
    )


def apply_pbc_nbonds(*, nbxmod: int = 5, cutnb: float = 18.0) -> None:
    """Nonbond list for periodic CHARMM (``cutim >= cutnb``)."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import pbc_nbond_kwargs

    import pycharmm

    cutim = float(cutnb) + 4.0
    pycharmm.NonBondedScript(**pbc_nbond_kwargs(nbxmod=nbxmod, cutnb=cutnb, cutim=cutim)).run()


def setup_charmm_environment(
    *,
    use_pbc: bool,
    cubic_box_side_A: float | None,
    nbxmod: int = 5,
) -> dict[str, Any]:
    """Vacuum or PBC CHARMM environment before MLpot registration."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prepare_charmm_vacuum,
        setup_default_nbonds,
    )

    if use_pbc:
        if cubic_box_side_A is None or float(cubic_box_side_A) <= 0.0:
            raise ValueError("PBC requires a positive cubic box side (Å)")
        from mmml.interfaces.pycharmmInterface.import_pycharmm import disable_charmm_domdec

        disable_charmm_domdec()
        prepare_charmm_pbc(float(cubic_box_side_A))
        apply_pbc_nbonds(nbxmod=nbxmod)
        return {"pbc": True, "box_A": float(cubic_box_side_A)}
    prepare_charmm_vacuum()
    setup_default_nbonds(nbxmod=nbxmod)
    return {"pbc": False, "box_A": None}
