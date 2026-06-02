"""Bonded-MM recovery mini after bad conformations (e.g. post-heat)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    BondedMmMiniConfig,
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


def rewrite_dynamics_restart_from_current_state(
    restart_path: PathLike | None,
    *,
    write_unit: int = 92,
) -> None:
    """Overwrite a dynamics restart so flags/coords match the current BLOCK setup."""
    if restart_path is None:
        return
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    path = Path(restart_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    restart_file = pycharmm.CharmmFile(
        file_name=str(path),
        file_unit=write_unit,
        formatted=True,
        read_only=False,
    )
    try:
        pycharmm.lingo.charmm_script(f"write restart unit {write_unit}\n")
    finally:
        restart_file.close()


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
    )
    if verbose:
        msg = f"MM baseline GRMS: {baseline.grms_kcalmol_A:.4f} kcal/mol/Å"
        if baseline.internal_kcalmol is not None:
            msg += f", internal: {baseline.internal_kcalmol:.4f} kcal/mol"
        print(msg, flush=True)
    return baseline


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
) -> None:
    """If MM bonded GRMS rose above baseline, run bonded-only SD (BLOCK toggle only)."""
    if not getattr(args, "bonded_mm_mini", False):
        return
    raw = str(getattr(args, "bonded_mm_mini_after", "heat") or "heat")
    watch = {s.strip().lower() for s in raw.split(",") if s.strip()}
    if stage.lower() not in watch:
        return
    if baseline is None:
        if not args.quiet:
            print(
                "bonded-MM-mini: no baseline strain recorded; skipping check",
                flush=True,
            )
        return

    grms_margin = getattr(args, "bonded_mm_grms_margin", None)
    margin = float(grms_margin if grms_margin is not None else getattr(args, "bonded_mm_internal_margin", 0.0))
    current_grms = measure_mm_grms_with_full_block(ctx)
    threshold = float(baseline.grms_kcalmol_A) + margin
    if current_grms <= threshold:
        if not args.quiet:
            msg = (
                f"bonded-MM-mini: GRMS {current_grms:.4f} kcal/mol/Å OK after {stage} "
                f"(baseline {baseline.grms_kcalmol_A:.4f} + margin {margin:.4f})"
            )
            if baseline.internal_kcalmol is not None:
                msg += f"; baseline internal {baseline.internal_kcalmol:.4f} kcal/mol"
            print(msg, flush=True)
    else:
        if not args.quiet:
            print(
                f"bonded-MM-mini: GRMS {current_grms:.4f} > {threshold:.4f} after {stage}; "
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

    rewrite_dynamics_restart_from_current_state(restart_path)
    if restart_path is not None and not args.quiet:
        print(
            f"bonded-MM-mini: resynced restart {Path(restart_path).name} "
            f"with current MLpot BLOCK",
            flush=True,
        )
