"""CHARMM nonbond presets shared by ``md_pbc_suite/ase.py`` and MLpot workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Vacuum cluster minimization (_run_charmm_minimize without PBC in ase.py).
VACUUM_CUTNB = 18.0
VACUUM_CTONNB = 13.0
VACUUM_CTOFNB = 17.0

# setupBox / legacy script string (PBC boxes).
PBC_CUTNB = 14.0
PBC_CTONNB = 10.0
PBC_CTOFNB = 12.0


def vacuum_nbond_kwargs(
    *,
    nbxmod: int = 5,
    cutnb: float = VACUUM_CUTNB,
    ctonnb: float = VACUUM_CTONNB,
    ctofnb: float = VACUUM_CTOFNB,
) -> dict[str, Any]:
    """Keyword args for ``pycharmm.NonBondedScript`` (vacuum, no crystal/image)."""
    return {
        "cutnb": float(cutnb),
        "ctonnb": float(ctonnb),
        "ctofnb": float(ctofnb),
        "eps": 1.0,
        "cdie": True,
        "atom": True,
        "vatom": True,
        "fswitch": True,
        "vfswitch": True,
        "nbxmod": int(nbxmod),
    }


def pbc_nbond_kwargs(
    *,
    nbxmod: int = 5,
    cutnb: float = VACUUM_CUTNB,
    cutim: float | None = None,
    ctonnb: float = VACUUM_CTONNB,
    ctofnb: float = VACUUM_CTOFNB,
) -> dict[str, Any]:
    """ASE PBC minimization: vacuum switched cutoffs plus ``cutim`` when periodic."""
    kw = vacuum_nbond_kwargs(
        nbxmod=nbxmod,
        cutnb=cutnb,
        ctonnb=ctonnb,
        ctofnb=ctofnb,
    )
    if cutim is not None:
        kw["cutim"] = float(cutim)
    # setupBox / crystal IMAGE scripts
    kw["inbfrq"] = -1
    kw["imgfrq"] = -1
    return kw


def read_cgenff_toppar(*, enable_drude: bool = False) -> None:
    """Load CGENFF RTF/PRM under relaxed BOMBlev; restore the prior level on exit."""
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        CGENFF_RTF,
        charmm_relaxed_bomlev,
    )

    with charmm_relaxed_bomlev():
        if enable_drude:
            read.rtf(CGENFF_RTF)
        else:
            read.rtf(_rtf_path_without_drude_autogen(CGENFF_RTF))
        read.prm(CGENFF_PRM)


def _rtf_path_without_drude_autogen(rtf_path: str | Path) -> str:
    """Return a temp RTF path with ``AUTO ... DRUDE`` removed (vacuum CGENFF MM only)."""
    import tempfile

    text = Path(rtf_path).read_text(encoding="utf-8", errors="replace")
    marker = "AUTO ANGLES DIHE PATCH DRUDE"
    if marker in text:
        text = text.replace(marker, "AUTO ANGLES DIHE PATCH", 1)
    fd, path = tempfile.mkstemp(suffix=".rtf", prefix="cgenff_no_drude_")
    import os

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def apply_vacuum_nbonds(*, nbxmod: int = 5) -> None:
    """Apply ASE-style vacuum nonbonds (domdec off, no crystal)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    import pycharmm

    prepare_charmm_vacuum()
    pycharmm.NonBondedScript(**vacuum_nbond_kwargs(nbxmod=nbxmod)).run()
