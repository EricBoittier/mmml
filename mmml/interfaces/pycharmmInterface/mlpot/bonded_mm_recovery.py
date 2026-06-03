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


@dataclass(frozen=True)
class BondedMmRecoveryResult:
    """Outcome of a bonded-MM recovery check."""

    ran_recovery: bool
    current: MmStrainBaseline | None = None
    reasons: tuple[str, ...] = ()
    heavy_reload: bool = False


def rewrite_dynamics_restart_from_current_state(
    restart_path: PathLike | None,
    *,
    write_unit: int = 92,
) -> None:
    """Write a ``WRITe restart`` snapshot from the current in-memory CHARMM state.

    Used before NPT stages after ``memory_handoff`` (e.g. post-heat bonded-MM mini)
    so overlap-chunk ``READYN`` has a valid restart (coords, velocities, crystal).
    """
    if restart_path is None:
        return
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    path = Path(restart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script(
            f"open write form unit {int(write_unit)} name {path}\n"
            f"write restart unit {int(write_unit)}\n"
            f"close unit {int(write_unit)}\n"
        )


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
    """GRMS + bonded internal + ANGL after ``ENER`` with full MM block (MLpot detached)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _with_mlpot_detached

    def _measure() -> MmStrainBaseline:
        apply_charmm_mm_block()
        pycharmm.lingo.charmm_script("ENER")
        return MmStrainBaseline(
            grms_kcalmol_A=float(charmm_grms()),
            internal_kcalmol=charmm_internal_energy_kcalmol(),
            angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
        )

    return _with_mlpot_detached(ctx, _measure)


def _bonded_strain_margins(args: argparse.Namespace) -> tuple[float, float, float]:
    grms_margin = float(
        getattr(args, "bonded_mm_grms_margin", None)
        if getattr(args, "bonded_mm_grms_margin", None) is not None
        else getattr(args, "bonded_mm_internal_margin", 0.0)
    )
    internal_margin = float(getattr(args, "bonded_mm_internal_energy_margin", 0.0) or 0.0)
    angl_margin = float(getattr(args, "bonded_mm_angl_margin", 0.0) or 0.0)
    return grms_margin, internal_margin, angl_margin


def _mlpot_covers_all_atoms(ctx: MlpotContext) -> bool:
    """True when CHARMM MM terms cannot be used as a recovery potential."""
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm

        n_atoms = int(pycharmm.coor.get_natom())
        n_ml = len(ctx.ml_selection.get_atom_indexes()) if ctx.ml_selection is not None else 0
        return n_atoms > 0 and n_ml >= n_atoms
    except Exception:
        return False


def _copy_mlpot_context_state(dst: MlpotContext, src: MlpotContext) -> None:
    """Keep the caller's context object valid after rebuilding CHARMM topology."""
    for name in (
        "mlpot",
        "pyCModel",
        "params",
        "model",
        "ml_selection",
        "block_tag",
        "ml_Z",
        "use_pbc",
        "cubic_box_side_A",
        "ml_charge",
        "ml_fq",
    ):
        setattr(dst, name, getattr(src, name))


def _measure_current_mm_strain() -> MmStrainBaseline:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

    apply_charmm_mm_block()
    pycharmm.lingo.charmm_script("ENER")
    return MmStrainBaseline(
        grms_kcalmol_A=float(charmm_grms()),
        internal_kcalmol=charmm_internal_energy_kcalmol(),
        angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
    )


def _reload_pre_mlpot_topology(
    ctx: MlpotContext,
    *,
    topology_psf: PathLike,
) -> None:
    """Replace MLpot-mutated PSF with the saved pre-MLpot topology."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    current_positions = get_charmm_positions_array()
    ctx.unset()
    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script(
            "DELETE ATOM SELE ALL END\nDELETE PSF SELE ALL END"
        )
        read.psf_card(str(Path(topology_psf).expanduser().resolve()))
    sync_charmm_positions(current_positions)
    if ctx.use_pbc and ctx.cubic_box_side_A is not None:
        setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(ctx.cubic_box_side_A))
    else:
        setup_default_nbonds()


def _reregister_mlpot_after_topology_reload(ctx: MlpotContext) -> None:
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        refresh_nbonds_after_mlpot_pbc,
        register_mlpot,
        select_all_atoms,
        sync_charmm_positions,
    )

    positions = get_charmm_positions_array()
    ml_z = np.asarray(ctx.ml_Z, dtype=int)
    new_ctx = register_mlpot(
        ctx.pyCModel,
        ml_z,
        select_all_atoms(),
        ml_charge=ctx.ml_charge,
        ml_fq=ctx.ml_fq,
        use_pbc=ctx.use_pbc,
    )
    new_ctx.ml_Z = ml_z
    new_ctx.use_pbc = bool(ctx.use_pbc)
    new_ctx.cubic_box_side_A = ctx.cubic_box_side_A
    if new_ctx.use_pbc and new_ctx.cubic_box_side_A is not None:
        refresh_nbonds_after_mlpot_pbc(
            cubic_box_side_A=float(new_ctx.cubic_box_side_A),
            force=True,
        )
    sync_charmm_positions(positions)
    _copy_mlpot_context_state(ctx, new_ctx)


def _run_bonded_sd_without_mlpot(
    ctx: MlpotContext,
    config: BondedMmMiniConfig,
) -> float | None:
    """Run bonded-only SD after the pre-MLpot PSF has already been reloaded."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_mm_only_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _bonded_recovery_sd_kwargs,
        _import_pycharmm_modules,
    )

    apply_bonded_mm_only_block()
    pycharmm, cons_fix, *_ = _import_pycharmm_modules()
    minimize = _import_pycharmm_modules()[3]
    pycharmm.lingo.charmm_script("ENER")
    grms_before = float(charmm_grms())
    angl_before = charmm_bonded_term_kcalmol("ANGL")
    if config.verbose:
        msg = (
            f"Bonded-MM heavy mini start: GRMS={grms_before:.4f} kcal/mol/Å "
            "(pre-MLpot topology, MLpot detached)"
        )
        if angl_before is not None:
            msg += f", ANGL={angl_before:.4f} kcal/mol"
        print(msg, flush=True)
    if config.nstep_sd > 0:
        minimize.run_sd(**_bonded_recovery_sd_kwargs(ctx, config))
    pycharmm.lingo.charmm_script("ENER")
    grms_after = float(charmm_grms())
    if config.verbose:
        msg = f"Bonded-MM heavy mini end: GRMS={grms_after:.4f} kcal/mol/Å"
        angl_after = charmm_bonded_term_kcalmol("ANGL")
        if angl_after is not None:
            msg += f", ANGL={angl_after:.4f} kcal/mol"
        print(msg, flush=True)
    cons_fix.turn_off()
    return grms_after


def _run_heavy_bonded_recovery_check(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
    baseline: MmStrainBaseline,
    topology_psf: PathLike,
) -> BondedMmRecoveryResult:
    """Reload the clean topology, check real MM strain, optionally recover, and reattach MLpot."""
    path = Path(topology_psf).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pre-MLpot topology PSF not found: {path}")

    if not args.quiet:
        print(
            f"bonded-MM-mini: reloading pre-MLpot topology for all-ML recovery: {path.name}",
            flush=True,
        )
    _reload_pre_mlpot_topology(ctx, topology_psf=path)
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.restraints import clear_mmfp_restraints

        clear_mmfp_restraints()
        current = _measure_current_mm_strain()
        grms_margin, internal_margin, angl_margin = _bonded_strain_margins(args)
        reasons = tuple(
            _recovery_reasons(
                current,
                baseline,
                grms_margin=grms_margin,
                internal_margin=internal_margin,
                angl_margin=angl_margin,
            )
        )
        if not reasons:
            if not args.quiet:
                print(
                    f"bonded-MM-mini: strain OK after {stage} with reloaded topology "
                    f"(GRMS {current.grms_kcalmol_A:.4f} kcal/mol/Å)",
                    flush=True,
                )
            return BondedMmRecoveryResult(
                ran_recovery=False,
                current=current,
                reasons=reasons,
                heavy_reload=True,
            )

        if not args.quiet:
            print(
                f"bonded-MM-mini: after {stage}: {'; '.join(reasons)}; "
                "running heavy bonded SD (pre-MLpot topology)",
                flush=True,
            )
        nstep = int(getattr(args, "bonded_mm_mini_steps", 50))
        _run_bonded_sd_without_mlpot(
            ctx,
            BondedMmMiniConfig(
                nstep_sd=nstep,
                nprint=max(1, int(getattr(args, "dyn_nprint", 500))),
                verbose=not args.quiet,
                show_energy=bool(getattr(args, "show_energy", False)),
            ),
        )
        return BondedMmRecoveryResult(
            ran_recovery=True,
            current=current,
            reasons=reasons,
            heavy_reload=True,
        )
    finally:
        _reregister_mlpot_after_topology_reload(ctx)
        _restore_flat_bottom_after_heavy_recovery(args)


def _restore_flat_bottom_after_heavy_recovery(args: argparse.Namespace) -> None:
    """Reinstall the workflow MMFP wall after temporarily clearing it for MM recovery."""
    fb_rad = getattr(args, "fb_rad", None)
    if fb_rad is None or float(fb_rad) <= 0:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_flat_bottom_selection,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        FlatBottomSphereConfig,
        setup_flat_bottom_sphere_mmfp,
    )

    setup_flat_bottom_sphere_mmfp(
        FlatBottomSphereConfig(
            radius=float(fb_rad),
            force=float(getattr(args, "fb_forc", 1.0)),
            xref=float(getattr(args, "fb_xref", 0.0)),
            yref=float(getattr(args, "fb_yref", 0.0)),
            zref=float(getattr(args, "fb_zref", 0.0)),
            selection=resolve_flat_bottom_selection(args),
        )
    )


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
    topology_psf: PathLike | None = None,
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

    if _mlpot_covers_all_atoms(ctx):
        if topology_psf is not None:
            result = _run_heavy_bonded_recovery_check(
                ctx,
                args,
                stage=stage,
                baseline=baseline,
                topology_psf=topology_psf,
            )
            return result.ran_recovery
        if not args.quiet:
            print(
                f"bonded-MM-mini: skipping after {stage} for all-ML system; "
                "CHARMM bonded terms are unavailable after MLpot registration "
                "and no pre-MLpot topology PSF was provided",
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
            f"running bonded SD (MLpot detached)",
            flush=True,
        )
    nstep = int(getattr(args, "bonded_mm_mini_steps", 50))
    minimize_bonded_mm_recovery(
        ctx,
        BondedMmMiniConfig(
            nstep_sd=nstep,
            nprint=max(1, int(getattr(args, "dyn_nprint", 500))),
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
