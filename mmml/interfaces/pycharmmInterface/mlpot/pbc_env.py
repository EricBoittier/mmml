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


def get_charmm_cubic_box_side_A() -> float:
    """Read the current cubic CHARMM cell side length (Å) from ``pbound_get_size``."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm.lib as lib

    is_cubic = lib.charmm.pbound_is_cubic_box()
    if _charmm_ctypes_scalar(is_cubic) != 1:
        raise RuntimeError("CHARMM box is not cubic; MLpot MIC sync expects a cubic cell")
    size_x = ctypes.c_double(0.0)
    size_y = ctypes.c_double(0.0)
    size_z = ctypes.c_double(0.0)
    lib.charmm.pbound_get_size(
        ctypes.byref(size_x),
        ctypes.byref(size_y),
        ctypes.byref(size_z),
    )
    lx = _charmm_ctypes_scalar(size_x)
    ly = _charmm_ctypes_scalar(size_y)
    lz = _charmm_ctypes_scalar(size_z)
    if min(lx, ly, lz) <= 0.0:
        raise RuntimeError(
            f"CHARMM cubic box side must be > 0, got ({lx}, {ly}, {lz})"
        )
    return lx


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
