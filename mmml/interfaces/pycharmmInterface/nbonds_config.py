"""CHARMM nonbond presets shared by ``md_pbc_suite/ase.py`` and MLpot workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Vacuum cluster minimization (_run_charmm_minimize without PBC in ase.py).
VACUUM_CUTNB = 18.0
VACUUM_CTONNB = 13.0
VACUUM_CTOFNB = 17.0
VACUUM_CUTIM_OFFSET = 4.0

# setupBox / legacy script string (PBC boxes).
PBC_CUTNB = 14.0
PBC_CTONNB = 10.0
PBC_CTOFNB = 12.0

# CHARMM requires primary/image cutoffs strictly below half the cubic box side.
PBC_NBOND_BOX_MARGIN_A = 1.0
PBC_NBOND_MIN_CUTNB_A = 6.0


@dataclass(frozen=True)
class PbcNbondCutoffs:
    """Nonbond cutoffs scaled to fit a cubic periodic box."""

    cubic_box_side_A: float
    cutnb: float
    cutim: float
    ctonnb: float
    ctofnb: float
    ctexnb: float

    @property
    def was_capped(self) -> bool:
        """True when any cutoff is below the large-box vacuum preset."""
        ideal_cutim = float(VACUUM_CUTNB) + float(VACUUM_CUTIM_OFFSET)
        return (
            self.cutnb < float(VACUUM_CUTNB) - 1e-6
            or self.cutim < ideal_cutim - 1e-6
            or self.ctonnb < float(VACUUM_CTONNB) - 1e-6
            or self.ctofnb < float(VACUUM_CTOFNB) - 1e-6
        )

    def summary_line(self) -> str:
        return (
            f"PBC nbonds capped for cubic L={self.cubic_box_side_A:.2f} Å: "
            f"cutnb/cutim={self.cutnb:.2f}/{self.cutim:.2f} Å, "
            f"ctonnb/ctofnb={self.ctonnb:.2f}/{self.ctofnb:.2f} Å, "
            f"ctexnb={self.ctexnb:.2f} Å "
            "(CHARMM requires cutoffs < L/2)"
        )

    def charm_ctexnb_A(self) -> float:
        """``ctexnb`` passed to CHARMM (may bump by 1 Å when the box allows).

        CHARMM can segfault when ``ctexnb`` equals ``cutnb`` exactly on some builds.
        Bump by 1 Å only when that stays strictly below ``L/2``.
        """
        return charm_ctexnb_A(
            self.ctexnb,
            cubic_box_side_A=self.cubic_box_side_A,
        )

    def as_pbc_nbond_kwargs(self, *, nbxmod: int = 1) -> dict[str, Any]:
        return pbc_nbond_kwargs(
            nbxmod=nbxmod,
            cutnb=self.cutnb,
            cutim=self.cutim,
            ctonnb=self.ctonnb,
            ctofnb=self.ctofnb,
            ctexnb=self.charm_ctexnb_A(),
        )


def charm_ctexnb_A(
    ctexnb: float,
    *,
    cubic_box_side_A: float,
    bump_A: float = 1.0,
) -> float:
    """Return PBC-safe ``ctexnb`` for CHARMM, with optional +1 Å bump when allowed."""
    base = float(ctexnb)
    bumped = base + float(bump_A)
    L = float(cubic_box_side_A)
    if not (L > 0.0) or L != L:  # nan / invalid box: vacuum-style preset
        return bumped
    half = 0.5 * L
    if bumped < half:
        return bumped
    return base


def scale_vacuum_switch_cutoffs(
    cutnb: float,
    *,
    cutnb_ref: float = VACUUM_CUTNB,
    ctonnb_ref: float = VACUUM_CTONNB,
    ctofnb_ref: float = VACUUM_CTOFNB,
) -> tuple[float, float]:
    """Scale switched VDW cutoffs with ``cutnb``; preserve ``ctonnb < ctofnb < cutnb``."""
    ref = float(cutnb_ref)
    nb = float(cutnb)
    scale = nb / ref if ref > 0.0 else 1.0
    ctonnb = float(ctonnb_ref) * scale
    ctofnb = float(ctofnb_ref) * scale
    ctonnb = min(ctonnb, nb - 4.0)
    ctofnb = min(ctofnb, nb - 0.5)
    ctofnb = max(ctofnb, ctonnb + 0.5)
    ctonnb = max(float(PBC_NBOND_MIN_CUTNB_A), min(ctonnb, ctofnb - 0.5))
    ctofnb = max(ctonnb + 0.5, min(ctofnb, nb - 0.5))
    return ctonnb, ctofnb


def pbc_nbond_cutoffs(
    cubic_box_side_A: float,
    *,
    cutnb_max: float = VACUUM_CUTNB,
    ctonnb_max: float = VACUUM_CTONNB,
    ctofnb_max: float = VACUUM_CTOFNB,
    margin_A: float = PBC_NBOND_BOX_MARGIN_A,
) -> PbcNbondCutoffs:
    """Return PBC-safe ``cutnb``/``cutim``/switch cutoffs for a cubic cell.

    Defaults like ``cutnb=18`` Å are invalid when ``L/2`` is too small and can
    segfault in CHARMM ``upinb`` during MLpot registration.
    """
    L = float(cubic_box_side_A)
    if L <= 0.0:
        raise ValueError(f"cubic box side must be > 0, got {L}")
    half = 0.5 * L
    max_cut = max(float(PBC_NBOND_MIN_CUTNB_A), half - float(margin_A))

    cutnb = min(float(cutnb_max), max_cut)
    ideal_cutim = float(cutnb_max) + float(VACUUM_CUTIM_OFFSET)
    cutim = min(ideal_cutim, max_cut)
    cutim = max(cutim, cutnb)

    ctonnb, ctofnb = scale_vacuum_switch_cutoffs(
        cutnb,
        cutnb_ref=float(cutnb_max),
        ctonnb_ref=float(ctonnb_max),
        ctofnb_ref=float(ctofnb_max),
    )
    return PbcNbondCutoffs(
        cubic_box_side_A=L,
        cutnb=cutnb,
        cutim=cutim,
        ctonnb=ctonnb,
        ctofnb=ctofnb,
        ctexnb=cutnb,
    )


def vacuum_nbond_kwargs(
    *,
    nbxmod: int = 1,
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
    nbxmod: int = 1,
    cutnb: float = VACUUM_CUTNB,
    cutim: float | None = None,
    ctonnb: float = VACUUM_CTONNB,
    ctofnb: float = VACUUM_CTOFNB,
    ctexnb: float | None = None,
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
    if ctexnb is not None:
        kw["ctexnb"] = float(ctexnb)
    # setupBox / crystal IMAGE scripts
    kw["inbfrq"] = 50
    kw["imgfrq"] = 50
    return kw


def read_cgenff_prm(
    prm_path: str | Path | None = None,
    *,
    append: bool = False,
    bomlev: bool = True,
) -> None:
    """Read bundled CGENFF ``.prm`` with FLEX (required for append overrides).

    All CGENFF parameter reads must use ``flex=True`` so later ``append=True``
    swaps (zeroed/full MLpot registration) do not trigger PARMIO level -2.
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    path = str(prm_path or CGENFF_PRM)

    def _read() -> None:
        read.prm(path, append=append, flex=True)

    if bomlev:
        with charmm_relaxed_bomlev():
            _read()
    else:
        _read()


def read_cgenff_toppar(*, enable_drude: bool = False) -> None:
    """Load CGENFF RTF/PRM under relaxed BOMBlev; restore the prior level on exit."""
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF

    with charmm_relaxed_bomlev():
        if enable_drude:
            read.rtf(CGENFF_RTF)
        else:
            read.rtf(_rtf_path_without_drude_autogen(CGENFF_RTF))
        read_cgenff_prm(bomlev=False)


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
