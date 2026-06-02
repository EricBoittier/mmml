"""Bonded-MM recovery mini after bad conformations (e.g. post-heat)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    BondedMmMiniConfig,
    charmm_internal_energy_kcalmol,
    measure_mm_internal_with_full_block,
    minimize_bonded_mm_recovery,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


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


def record_mm_baseline_internal_energy(
    *,
    verbose: bool = False,
) -> float | None:
    """Record CHARMM internal energy after the first MM-only pre-minimize."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block

    apply_charmm_mm_block()
    pycharmm.lingo.charmm_script("ENER")
    baseline = charmm_internal_energy_kcalmol()
    if verbose and baseline is not None:
        print(
            f"MM baseline internal energy: {baseline:.4f} kcal/mol",
            flush=True,
        )
    return baseline


def maybe_run_bonded_mm_mini_after_stage(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
    baseline_internal: float | None,
    restart_path: PathLike | None = None,
) -> None:
    """If internal energy rose above baseline, run bonded-only MM SD and restore BLOCK."""
    if not getattr(args, "bonded_mm_mini", False):
        return
    raw = str(getattr(args, "bonded_mm_mini_after", "heat") or "heat")
    watch = {s.strip().lower() for s in raw.split(",") if s.strip()}
    if stage.lower() not in watch:
        return
    if baseline_internal is None:
        if not args.quiet:
            print(
                "bonded-MM-mini: no baseline internal energy recorded; skipping check",
                flush=True,
            )
        return

    margin = float(getattr(args, "bonded_mm_internal_margin", 0.0))
    current = measure_mm_internal_with_full_block(ctx)
    threshold = float(baseline_internal) + margin
    if current <= threshold:
        if not args.quiet:
            print(
                f"bonded-MM-mini: internal {current:.4f} kcal/mol OK after {stage} "
                f"(baseline {baseline_internal:.4f} + margin {margin:.4f})",
                flush=True,
            )
    else:
        if not args.quiet:
            print(
                f"bonded-MM-mini: internal {current:.4f} > {threshold:.4f} after {stage}; "
                f"running bonded SD",
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

    # Internal check temporarily detaches MLpot / toggles BLOCK; refresh restart flags.
    rewrite_dynamics_restart_from_current_state(restart_path)
    if restart_path is not None and not args.quiet:
        print(
            f"bonded-MM-mini: resynced restart {Path(restart_path).name} "
            f"with current MLpot BLOCK",
            flush=True,
        )
