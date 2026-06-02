"""Bonded-MM recovery mini after bad conformations (e.g. post-heat)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    BondedMmMiniConfig,
    charmm_bonded_term_kcalmol,
    charmm_internal_energy_kcalmol,
    measure_mm_grms_with_full_block,
    minimize_bonded_mm_recovery,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


@dataclass(frozen=True)
class MmStrainBaseline:
    """Reference MM strain after the first CHARMM MM-only pre-minimize."""

    grms_kcalmol_A: float
    internal_kcalmol: float | None = None
    angl_kcalmol: float | None = None


def rewrite_dynamics_restart_from_current_state(
    restart_path: PathLike | None,
    *,
    write_unit: int = 92,
) -> None:
    """No-op: dynamics restart files are only written by ``WRIDYN`` during ``dyna``.

    Same-process stage handoffs use in-memory coordinates (see ``memory_handoff``).
    """
    _ = (restart_path, write_unit)


def record_mm_baseline_strain(*, verbose: bool = False) -> MmStrainBaseline | None:
    """Record MM GRMS (+ internal energy when readable) after MM-only pre-minimize."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

    apply_charmm_mm_block()
    pycharmm.lingo.charmm_script("ENER")
    baseline = MmStrainBaseline(
        grms_kcalmol_A=float(charmm_grms()),
        internal_kcalmol=charmm_internal_energy_kcalmol(),
        angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
    )
    if verbose:
        msg = f"MM baseline GRMS: {baseline.grms_kcalmol_A:.4f} kcal/mol/Å"
        if baseline.angl_kcalmol is not None:
            msg += f", ANGL: {baseline.angl_kcalmol:.4f} kcal/mol"
        if baseline.internal_kcalmol is not None:
            msg += f", internal: {baseline.internal_kcalmol:.4f} kcal/mol"
        print(msg, flush=True)
    return baseline


def measure_mm_bonded_strain_with_full_block(ctx: MlpotContext) -> MmStrainBaseline:
    """GRMS + bonded internal + ANGL after ``ENER`` with full MM block (MLpot on)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _with_mlpot_block_restored

    def _measure() -> MmStrainBaseline:
        apply_charmm_mm_block()
        pycharmm.lingo.charmm_script("ENER")
        return MmStrainBaseline(
            grms_kcalmol_A=float(charmm_grms()),
            internal_kcalmol=charmm_internal_energy_kcalmol(),
            angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
        )

    return _with_mlpot_block_restored(ctx, _measure)


def _bonded_strain_margins(args: argparse.Namespace) -> tuple[float, float, float]:
    grms_margin = float(
        getattr(args, "bonded_mm_grms_margin", None)
        if getattr(args, "bonded_mm_grms_margin", None) is not None
        else getattr(args, "bonded_mm_internal_margin", 0.0)
    )
    internal_margin = float(getattr(args, "bonded_mm_internal_energy_margin", 0.0) or 0.0)
    angl_margin = float(getattr(args, "bonded_mm_angl_margin", 0.0) or 0.0)
    return grms_margin, internal_margin, angl_margin


def _recovery_reasons(
    current: MmStrainBaseline,
    baseline: MmStrainBaseline,
    *,
    grms_margin: float,
    internal_margin: float,
    angl_margin: float,
) -> list[str]:
    reasons: list[str] = []
    grms_thr = float(baseline.grms_kcalmol_A) + grms_margin
    if current.grms_kcalmol_A > grms_thr:
        reasons.append(
            f"GRMS {current.grms_kcalmol_A:.4f} > {grms_thr:.4f} kcal/mol/Å"
        )
    if (
        baseline.internal_kcalmol is not None
        and current.internal_kcalmol is not None
        and internal_margin > 0.0
    ):
        int_thr = float(baseline.internal_kcalmol) + internal_margin
        if current.internal_kcalmol > int_thr:
            reasons.append(
                f"internal {current.internal_kcalmol:.4f} > {int_thr:.4f} kcal/mol"
            )
    if (
        baseline.angl_kcalmol is not None
        and current.angl_kcalmol is not None
        and angl_margin > 0.0
    ):
        angl_thr = float(baseline.angl_kcalmol) + angl_margin
        if current.angl_kcalmol > angl_thr:
            reasons.append(
                f"ANGL {current.angl_kcalmol:.4f} > {angl_thr:.4f} kcal/mol"
            )
    return reasons


def assert_pre_min_bonded_geometry(
    args: argparse.Namespace,
    *,
    baseline: MmStrainBaseline | None = None,
) -> None:
    """Abort (unless allowed) when Packmol/MM pre-min leaves high ANGL or internal strain."""
    import os
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block

    if not getattr(args, "bonded_mm_mini", False):
        return

    apply_charmm_mm_block()
    import pycharmm

    pycharmm.lingo.charmm_script("ENER")
    angl = charmm_bonded_term_kcalmol("ANGL")
    internal = charmm_internal_energy_kcalmol()
    max_angl = getattr(args, "bonded_mm_max_angl_kcal", None)
    max_internal = getattr(args, "bonded_mm_max_internal_kcal", None)
    problems: list[str] = []
    if max_angl is not None and angl is not None and float(angl) > float(max_angl):
        problems.append(f"ANGL {float(angl):.4f} > limit {float(max_angl):.4f} kcal/mol")
    if (
        max_internal is not None
        and internal is not None
        and float(internal) > float(max_internal)
    ):
        problems.append(
            f"internal {float(internal):.4f} > limit {float(max_internal):.4f} kcal/mol"
        )
    if not problems:
        if not args.quiet and angl is not None:
            msg = f"Pre-min bonded geometry: ANGL={float(angl):.4f} kcal/mol"
            if internal is not None:
                msg += f", internal={float(internal):.4f} kcal/mol"
            if baseline is not None:
                msg += f" (baseline GRMS {baseline.grms_kcalmol_A:.4f} kcal/mol/Å)"
            print(msg, flush=True)
        return

    allow = bool(getattr(args, "allow_high_bonded_strain", False)) or (
        (os.environ.get("MMML_MLPOT_ALLOW_HIGH_BONDED_STRAIN") or "")
        .strip()
        .lower()
        in ("1", "yes", "true")
    )
    text = (
        "Pre-min bonded strain limits exceeded after CHARMM MM minimize:\n  "
        + "\n  ".join(problems)
        + "\nIncrease --charmm-sd-steps / --charmm-abnr-steps, relax Packmol, "
        "or pass --allow-high-bonded-strain (not recommended)."
    )
    if allow:
        print(f"WARN: {text}", file=sys.stderr, flush=True)
        return
    print(text, file=sys.stderr, flush=True)
    raise SystemExit(1)


def record_mm_baseline_internal_energy(
    *,
    verbose: bool = False,
) -> float | None:
    """Legacy helper: internal energy only (prefer :func:`record_mm_baseline_strain`)."""
    baseline = record_mm_baseline_strain(verbose=verbose)
    return baseline.internal_kcalmol if baseline is not None else None


def maybe_run_bonded_mm_mini_after_stage(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
    baseline: MmStrainBaseline | None,
    restart_path: PathLike | None = None,
) -> bool:
    """If MM bonded strain rose above baseline, run bonded-only SD (BLOCK toggle only).

    Triggers when GRMS exceeds baseline + ``--bonded-mm-grms-margin``, and/or (if margins
    > 0) internal or ANGL exceed baseline + their margins.

    Returns True when recovery SD ran (caller should hand off next stage from memory).
    """
    if not getattr(args, "bonded_mm_mini", False):
        return False
    raw = str(getattr(args, "bonded_mm_mini_after", "heat") or "heat")
    watch = {s.strip().lower() for s in raw.split(",") if s.strip()}
    if stage.lower() not in watch:
        return False
    if baseline is None:
        if not args.quiet:
            print(
                "bonded-MM-mini: no baseline strain recorded; skipping check",
                flush=True,
            )
        return False

    grms_margin, internal_margin, angl_margin = _bonded_strain_margins(args)
    current = measure_mm_bonded_strain_with_full_block(ctx)
    reasons = _recovery_reasons(
        current,
        baseline,
        grms_margin=grms_margin,
        internal_margin=internal_margin,
        angl_margin=angl_margin,
    )
    if not reasons:
        if not args.quiet:
            msg = f"bonded-MM-mini: strain OK after {stage} (GRMS {current.grms_kcalmol_A:.4f}"
            if current.angl_kcalmol is not None:
                msg += f", ANGL {current.angl_kcalmol:.4f}"
            if current.internal_kcalmol is not None:
                msg += f", internal {current.internal_kcalmol:.4f}"
            msg += " kcal/mol)"
            print(msg, flush=True)
        return False

    if not args.quiet:
        print(
            f"bonded-MM-mini: after {stage}: {'; '.join(reasons)}; "
            f"running bonded SD (MLpot stays registered)",
            flush=True,
        )
    nstep = int(getattr(args, "bonded_mm_mini_steps", 50))
    minimize_bonded_mm_recovery(
        ctx,
        BondedMmMiniConfig(
            nstep_sd=nstep,
            nprint=max(1, int(getattr(args, "dyn_nprint", 100))),
            verbose=not args.quiet,
            show_energy=bool(getattr(args, "show_energy", False)),
        ),
    )
    if not args.quiet:
        print(
            f"bonded-MM-mini: next stage will continue from in-memory coordinates "
            f"(restart file unchanged)",
            flush=True,
        )
    return True
