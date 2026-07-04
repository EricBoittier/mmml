"""CHARMM nonbond presets shared by ``md_pbc_suite/ase.py`` and MLpot workflows."""

from __future__ import annotations

import math
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
    _nb, ctonnb, ctofnb = enforce_switch_cutoff_order(nb, ctonnb, ctofnb)
    return ctonnb, ctofnb


def enforce_switch_cutoff_order(
    cutnb: float,
    ctonnb: float,
    ctofnb: float,
) -> tuple[float, float, float]:
    """Guarantee ``ctonnb < ctofnb < cutnb`` for CHARMM nbonds setters."""
    nb = float(cutnb)
    on = float(ctonnb)
    of = float(ctofnb)
    of = min(of, nb - 0.5)
    on = min(on, of - 0.5)
    on = max(float(PBC_NBOND_MIN_CUTNB_A), on)
    of = max(on + 0.5, of)
    nb = max(of + 0.5, nb)
    return nb, on, of


def mlpot_mm_nl_cutoff_A(
    *,
    mm_switch_on: float,
    mm_switch_width: float,
) -> float:
    """Outer MM neighbor-list radius used by JAX switched-MM (Å)."""
    return float(mm_switch_on) + float(mm_switch_width)


def pbc_nbond_cutoffs_from_mlpot_switches(
    cubic_box_side_A: float,
    *,
    mm_switch_on: float,
    mm_switch_width: float,
    margin_A: float = PBC_NBOND_BOX_MARGIN_A,
) -> PbcNbondCutoffs:
    """PBC CHARMM cutoffs aligned with MMML ``mm_switch_on + mm_switch_width``."""
    cutnb_max = mlpot_mm_nl_cutoff_A(
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
    )
    return pbc_nbond_cutoffs(
        cubic_box_side_A,
        cutnb_max=cutnb_max,
        ctonnb_max=float(ctonnb_max_for_cutnb(cutnb_max)),
        ctofnb_max=float(ctofnb_max_for_cutnb(cutnb_max)),
        margin_A=margin_A,
    )


def ctonnb_max_for_cutnb(cutnb: float) -> float:
    """Scale vacuum ``ctonnb`` reference to a target ``cutnb``."""
    ctonnb, _ = scale_vacuum_switch_cutoffs(float(cutnb))
    return ctonnb


def ctofnb_max_for_cutnb(cutnb: float) -> float:
    """Scale vacuum ``ctofnb`` reference to a target ``cutnb``."""
    _, ctofnb = scale_vacuum_switch_cutoffs(float(cutnb))
    return ctofnb


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
    cutnb, ctonnb, ctofnb = enforce_switch_cutoff_order(cutnb, ctonnb, ctofnb)
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


# CGENFF flex/append reads and IC prm_fill can emit PARRDR level -3; bomlev -2 aborts.
CGENFF_PRM_BOMLEV = -5


def suspend_pbc_before_cgenff_param_append() -> bool:
    """``crystal free`` before ``READ PARAM APPEND`` when periodic images are active.

    ``read_param_file`` clears IMAGE tables but leaves ``NTRANS > 0``, so the
  follow-up ``upinb`` → ``UPIMNB`` pass can segfault (MLpot registration,
    overlap-rescue ``reregister_params``, etc.).
    """
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        PYCHARMM_AVAILABLE,
        crystal_free_charmm_for_param_append,
    )

    if not PYCHARMM_AVAILABLE:
        return False

    def _suspend() -> bool:
        if not crystal_free_charmm_for_param_append():
            return False
        print(
            "MMML: crystal free before CGENFF READ PARAM APPEND (PBC suspend)",
            flush=True,
        )
        return True

    try:
        import pycharmm.image as image

        if int(image.get_ntrans()) > 1:
            return _suspend()
    except Exception:
        pass
    try:
        import ctypes

        import pycharmm.lib as lib

        sx = ctypes.c_double(0.0)
        sy = ctypes.c_double(0.0)
        sz = ctypes.c_double(0.0)
        lib.charmm.pbound_get_size(
            ctypes.byref(sx),
            ctypes.byref(sy),
            ctypes.byref(sz),
        )
        if min(float(sx.value), float(sy.value), float(sz.value)) > 0.0:
            return _suspend()
    except Exception:
        pass
    return False


def read_cgenff_prm(
    prm_path: str | Path | None = None,
    *,
    append: bool = False,
    bomlev: bool = True,
    bomlev_level: int | None = None,
) -> None:
    """Read bundled CGENFF ``.prm`` with FLEX (required for append overrides).

    All CGENFF parameter reads use ``flex=True`` and ``bomlev=-5`` by default
    (PARRDR/PARMIO warnings during flex load and append swaps).
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    path = str(prm_path or CGENFF_PRM)
    level = int(bomlev_level if bomlev_level is not None else CGENFF_PRM_BOMLEV)

    if append:
        suspend_pbc_before_cgenff_param_append()

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        charmm_quiet_prnlev,
        suppress_charmm_fortran_io,
    )

    def _read() -> None:
        with charmm_quiet_prnlev(), suppress_charmm_fortran_io():
            read.prm(path, append=append, flex=True)

    if bomlev:
        with charmm_relaxed_bomlev(level):
            _read()
    else:
        _read()


def ic_prm_fill(*, replace_all: bool = True) -> None:
    """Fill IC table from the parameter file under relaxed bomlev."""
    import pycharmm.ic as ic

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        ic.prm_fill(replace_all=replace_all)


def read_cgenff_toppar(*, enable_drude: bool = False) -> None:
    """Load CGENFF RTF/PRM under relaxed BOMBlev; restore the prior level on exit."""
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.charmm_paths import assert_cgenff_toppar_readable
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF

    toppar = assert_cgenff_toppar_readable()

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        if enable_drude:
            read.rtf(str(toppar.rtf))
        else:
            read.rtf(_rtf_path_without_drude_autogen(toppar.rtf))
        read_cgenff_prm(prm_path=toppar.prm, bomlev=False)


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


_PBC_NBOND_STASH_ATTR = "_pbc_nbond_cutoffs_applied"


def _pbc_box_L_match(stored_L: float, target_L: float) -> bool:
    return abs(float(stored_L) - float(target_L)) <= max(1e-3, 1e-4 * float(target_L))


def pbc_nbond_cutoffs_invariant_ok(
    cuts: PbcNbondCutoffs,
    *,
    strict_half: bool = True,
) -> bool:
    """True when ``0 < ctonnb < ctofnb <= cutnb < L/2``."""
    L = float(cuts.cubic_box_side_A)
    if not (L > 0.0) or L != L:
        return False
    if float(cuts.ctonnb) <= 0.0:
        return False
    if not charmm_nbond_cutoffs_orderly(cuts.cutnb, cuts.ctonnb, cuts.ctofnb):
        return False
    if float(cuts.ctofnb) > float(cuts.cutnb) + 1e-6:
        return False
    if strict_half and not (float(cuts.cutnb) < 0.5 * L):
        return False
    return True


def stash_pbc_nbond_cutoffs(workflow_args: Any, cuts: PbcNbondCutoffs) -> None:
    """Remember PBC nbond targets applied during pretreat for MLpot registration."""
    if workflow_args is None:
        return
    setattr(workflow_args, _PBC_NBOND_STASH_ATTR, cuts)


def get_stashed_pbc_nbond_cutoffs(
    workflow_args: Any,
    cubic_box_side_A: float,
) -> PbcNbondCutoffs | None:
    if workflow_args is None:
        return None
    cuts = getattr(workflow_args, _PBC_NBOND_STASH_ATTR, None)
    if not isinstance(cuts, PbcNbondCutoffs):
        return None
    if not _pbc_box_L_match(cuts.cubic_box_side_A, cubic_box_side_A):
        return None
    return cuts


def resolve_pbc_nbond_cutoffs(
    cubic_box_side_A: float,
    *,
    workflow_args: Any = None,
    cutoff_params: Any = None,
    mm_switch_on: float | None = None,
    mm_switch_width: float | None = None,
    prefer_stashed: bool = True,
    use_workflow_mm_switches: bool = False,
) -> PbcNbondCutoffs:
    """Single source of truth for PBC CHARMM nbond targets.

    Reuses cutoffs stashed during ``setup_charmm_environment`` / pretreat when the
    box side matches, so MLpot registration does not recompute uncapped vacuum
    switches after a tighter L/2 cap was already applied.

    Fresh targets use DEFAULT MM switches (8+5 Å) unless
    ``use_workflow_mm_switches=True`` or explicit ``mm_switch_on`` / width are
    passed.  Pretreat stashes the capped defaults; registration prefers that stash
    even when workflow JAX switches are wider (e.g. 12+6 Å).
    """
    L = float(cubic_box_side_A)
    if prefer_stashed:
        stashed = get_stashed_pbc_nbond_cutoffs(workflow_args, L)
        if stashed is not None and pbc_nbond_cutoffs_invariant_ok(stashed):
            return stashed

    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
        CutoffParameters,
        cutoff_parameters_from_args,
    )

    cp = cutoff_params
    if cp is None and workflow_args is not None and use_workflow_mm_switches:
        cp = cutoff_parameters_from_args(workflow_args)
    if cp is None:
        cp = CutoffParameters()

    if mm_switch_on is not None:
        mm_on = float(mm_switch_on)
    elif use_workflow_mm_switches:
        mm_on = float(getattr(cp, "mm_switch_on", DEFAULT_MM_SWITCH_ON))
    else:
        mm_on = float(DEFAULT_MM_SWITCH_ON)

    if mm_switch_width is not None:
        mm_w = float(mm_switch_width)
    elif use_workflow_mm_switches:
        mm_w = float(getattr(cp, "mm_switch_width", DEFAULT_MM_SWITCH_WIDTH))
    else:
        mm_w = float(DEFAULT_MM_SWITCH_WIDTH)

    cuts = pbc_nbond_cutoffs_from_mlpot_switches(
        L,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
    )
    if not pbc_nbond_cutoffs_invariant_ok(cuts):
        raise RuntimeError(
            f"PBC nbond cutoffs violate CHARMM ordering for L={L:.3f} Å: "
            f"cutnb/ctonnb/ctofnb={cuts.cutnb:.2f}/{cuts.ctonnb:.2f}/{cuts.ctofnb:.2f} "
            f"(require 0 < ctonnb < ctofnb <= cutnb < L/2={0.5 * L:.2f})"
        )
    return cuts


def assert_pbc_nbond_cutoffs_before_upinb(
    cuts: PbcNbondCutoffs,
    *,
    context: str = "PBC nbonds",
) -> None:
    """Raise when cutoffs would fail CHARMM ``upinb`` / ``UPDATE`` ordering checks."""
    if pbc_nbond_cutoffs_invariant_ok(cuts):
        return
    L = float(cuts.cubic_box_side_A)
    raise RuntimeError(
        f"{context}: invalid PBC nbond cutoffs for L={L:.3f} Å: "
        f"cutnb/ctonnb/ctofnb={cuts.cutnb:.2f}/{cuts.ctonnb:.2f}/{cuts.ctofnb:.2f} "
        f"(require 0 < ctonnb < ctofnb <= cutnb < L/2={0.5 * L:.2f})"
    )


def read_charmm_switch_cutoffs() -> tuple[float, float]:
    """Return live ``(ctonnb, ctofnb)`` from CHARMM (Å)."""
    import pycharmm.nbonds as nbonds

    return float(nbonds.get_ctonnb()), float(nbonds.get_ctofnb())


def try_get_charmm_cutnb() -> float:
    """Return live ``cutnb`` when ``nbonds_get_cutnb`` is linked; else NaN."""
    try:
        import pycharmm.lib as lib
        import pycharmm.nbonds as nbonds

        if not hasattr(lib.charmm, "nbonds_get_cutnb"):
            return float("nan")
        cutnb_fn = getattr(nbonds, "get_cutnb", None)
        if not callable(cutnb_fn):
            return float("nan")
        return float(cutnb_fn())
    except AttributeError:
        return float("nan")


def read_charmm_nbond_cutoffs() -> tuple[float, float, float]:
    """Return live ``(cutnb, ctonnb, ctofnb)`` from CHARMM (Å)."""
    ctonnb, ctofnb = read_charmm_switch_cutoffs()
    cutnb = try_get_charmm_cutnb()
    return cutnb, ctonnb, ctofnb


def _ensure_headroom_before_switch_apply(
    *,
    cutnb: float | None,
    ctonnb: float | None,
    ctofnb: float | None,
) -> None:
    """Raise ``cutnb`` when switch radii cannot be updated in CHARMM order.

    ``crystal.build`` on domdec builds can leave ``cutnb <= ctofnb`` with vacuum
    switches.  Raising ``cutnb`` before ``ctonnb``/``ctofnb`` updates avoids
    ``nbonds_cutoffs_orderly`` failures when lowering or raising switch radii.
    """
    import pycharmm.nbonds as nbonds

    live_on, live_of = read_charmm_switch_cutoffs()
    live_nb = try_get_charmm_cutnb()

    target_of = float(ctofnb) if ctofnb is not None else live_of
    target_on = float(ctonnb) if ctonnb is not None else live_on
    target_nb = (
        float(cutnb)
        if cutnb is not None
        else (live_nb if math.isfinite(live_nb) else float(VACUUM_CUTNB))
    )

    headroom = max(
        target_nb,
        target_of + 0.5,
        target_on + 1.0,
        live_of + 0.5,
        float(VACUUM_CUTNB),
    )
    if math.isfinite(live_nb):
        headroom = max(headroom, live_nb)

    needs_bump = not math.isfinite(live_nb) or live_nb <= max(live_of, target_of)
    if needs_bump:
        nbonds.set_cutnb(headroom)


def charmm_has_vacuum_nbond_preset(*, tol: float = 0.05) -> bool:
    """True when live switch cutoffs match the vacuum cluster preset."""
    ctonnb, ctofnb = read_charmm_switch_cutoffs()
    return (
        abs(ctonnb - float(VACUUM_CTONNB)) <= tol
        and abs(ctofnb - float(VACUUM_CTOFNB)) <= tol
    )


def charmm_nbond_cutoffs_match_target(
    cuts: PbcNbondCutoffs,
    *,
    tol: float = 0.05,
) -> bool:
    """True when live CHARMM cutoffs match capped PBC targets."""
    cutnb, ctonnb, ctofnb = read_charmm_nbond_cutoffs()
    if math.isfinite(cutnb):
        if abs(cutnb - float(cuts.cutnb)) > tol:
            return False
    return (
        charmm_nbond_cutoffs_orderly(float(cuts.cutnb), ctonnb, ctofnb)
        and abs(ctonnb - float(cuts.ctonnb)) <= tol
        and abs(ctofnb - float(cuts.ctofnb)) <= tol
    )


def log_charmm_pbc_nbond_cutoffs(
    cuts: PbcNbondCutoffs,
    *,
    context: str = "PBC nbonds",
    strict: bool = False,
) -> None:
    """Log live vs target PBC cutoffs; optionally raise when vacuum preset persists."""
    cutnb, ctonnb, ctofnb = read_charmm_nbond_cutoffs()
    target = (
        f"target cutnb/ctonnb/ctofnb="
        f"{cuts.cutnb:.2f}/{cuts.ctonnb:.2f}/{cuts.ctofnb:.2f} Å"
    )
    if math.isfinite(cutnb):
        live = f"live cutnb/ctonnb/ctofnb={cutnb:.2f}/{ctonnb:.2f}/{ctofnb:.2f} Å"
    else:
        live = f"live ctonnb/ctofnb={ctonnb:.2f}/{ctofnb:.2f} Å"
    ok = charmm_nbond_cutoffs_match_target(cuts)
    level = "OK" if ok else "MISMATCH"
    print(f"{context}: {level} ({live}; {target})", flush=True)
    if strict and not ok:
        raise RuntimeError(
            f"{context}: CHARMM nonbond cutoffs still at vacuum or out of range "
            f"({live}; {target}). Rebuild libcharmm if get_cutnb is unavailable."
        )


def charmm_nbond_cutoffs_orderly(
    cutnb: float,
    ctonnb: float,
    ctofnb: float,
) -> bool:
    """True when ``ctonnb < ctofnb < cutnb`` (CHARMM ``nbonds_cutoffs_orderly``)."""
    return float(ctonnb) < float(ctofnb) < float(cutnb)


def apply_switch_nbond_cutoffs_before_cutnb(
    ctonnb: float,
    ctofnb: float,
    *,
    rebuild: bool = False,
) -> None:
    """Lower switch radii before ``cutnb`` changes (``crystal.build`` on domdec builds)."""
    apply_nbonds_kwargs({"ctonnb": float(ctonnb), "ctofnb": float(ctofnb)}, rebuild=rebuild)


def trigger_nbonds_update_script() -> None:
    """Run CHARMM ``UPDATE`` without changing cutoff keywords."""
    import pycharmm.lingo as lingo

    lingo.charmm_script("UPDATE")


def apply_nbonds_kwargs(kw: dict[str, Any], *, rebuild: bool = True) -> None:
    """Apply nonbond settings via the KEY_LIBRARY C API (no ``nbonds`` script).

    ``libcharmm.so`` built with ``as_library=ON`` does not link the ``nbonds``
    parser command; use :mod:`pycharmm.nbonds` setters instead. ``nbxmod`` has no
    C setter — it is dropped here (CGENFF ``NONBONDED nbxmod`` from READ PARAM).

    When ``rebuild=False``, only cutoffs/frequencies are configured; callers must
  run ``update_bnbnd`` after PSF ML exclusions are installed (PBC registration).
    """
    import pycharmm.nbonds as nbonds

    cfg = dict(kw)
    cfg.pop("nbxmod", None)
    cutim = cfg.pop("cutim", None)
    inbfrq = cfg.pop("inbfrq", None)
    imgfrq = cfg.pop("imgfrq", None)
    cfg.pop("ctexnb", None)  # CHARMM bumps ctexnb internally when cutnb/cutim change

    if any(k in cfg for k in ("cutnb", "ctonnb", "ctofnb")):
        _ensure_headroom_before_switch_apply(
            cutnb=cfg.get("cutnb"),
            ctonnb=cfg.get("ctonnb"),
            ctofnb=cfg.get("ctofnb"),
        )

    # Apply switched cutoffs low-to-high (ctonnb, ctofnb, cutnb) so lowering ``cutnb``
    # never leaves stale switch values above the new primary cutoff.
    for key in ("ctonnb", "ctofnb", "cutnb"):
        if key in cfg:
            nbonds.configure(**{key: cfg.pop(key)})
    if cfg:
        nbonds.configure(**cfg)
    if cutim is not None:
        nbonds.set_cutim(float(cutim))
    if inbfrq is not None:
        nbonds.set_inbfrq(int(inbfrq))
    if imgfrq is not None:
        nbonds.set_imgfrq(int(imgfrq))
    if rebuild:
        nbonds.update_bnbnd()


def apply_nbonds_script_kwargs(kw: dict[str, Any], *, rebuild: bool = True) -> None:
    """Apply nonbond kwargs via the C API, then optionally ``UPDATE``.

    Do not pass cutoff kwargs to ``UpdateNonBondedScript`` — CHARMM parses
    ``cutnb`` before ``ctofnb``/``ctonnb`` and fails when lowering ``cutnb``.
    """
    apply_nbonds_kwargs(kw, rebuild=False)
    if rebuild:
        trigger_nbonds_update_script()


def apply_vacuum_nbonds(*, nbxmod: int = 5) -> None:
    """Apply ASE-style vacuum nonbonds (domdec off, no crystal)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    prepare_charmm_vacuum()
    apply_nbonds_kwargs(vacuum_nbond_kwargs(nbxmod=nbxmod))
