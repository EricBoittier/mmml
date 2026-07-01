"""Extent fly-off repack: in-memory reference geometry and post-repack polish."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

GeometryReferenceKind = Literal["baseline", "mini"]


def stash_geometry_reference_on_ctx(
    ctx: MlpotContext,
    *,
    kind: GeometryReferenceKind,
) -> None:
    """Keep a copy of current CHARMM coordinates for extent repack (no disk required)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    if pos is None or int(pos.size) == 0:
        return
    arr = np.asarray(pos, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return
    snapshot = arr.copy()
    if kind == "baseline":
        ctx.geometry_baseline_positions = snapshot
    else:
        ctx.geometry_mini_positions = snapshot


def _memory_extent_reference_sources(
    mlpot_ctx: MlpotContext | None,
) -> list[tuple[np.ndarray, Path]]:
    """Ordered in-memory templates: post-mini first, then pre-dynamics baseline."""
    if mlpot_ctx is None:
        return []
    out: list[tuple[np.ndarray, Path]] = []
    for attr, tag in (
        ("geometry_mini_positions", "in-memory-mini"),
        ("geometry_baseline_positions", "in-memory-baseline"),
    ):
        raw = getattr(mlpot_ctx, attr, None)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        out.append((arr, Path(f"<{tag}>")))
    return out


def in_memory_extent_reference_available(mlpot_ctx: MlpotContext | None) -> bool:
    return bool(_memory_extent_reference_sources(mlpot_ctx))


def resolve_extent_reference_positions(
    candidates: list[Path],
    mlpot_ctx: MlpotContext | None,
) -> tuple[np.ndarray, Path]:
    """Load repack template from disk ladder, then in-memory mini/baseline snapshots."""
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        _load_extent_reference_positions,
    )

    if candidates:
        try:
            return _load_extent_reference_positions(candidates)
        except RuntimeError:
            pass

    for arr, virtual in _memory_extent_reference_sources(mlpot_ctx):
        return arr, virtual

    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        same_residue_cluster_reference_from_ctx,
    )

    same_res = same_residue_cluster_reference_from_ctx(mlpot_ctx)
    if same_res is not None:
        return same_res, Path("<same-residue-cluster>")

    names = ", ".join(p.name for p in candidates) or "(none)"
    mem = "no" if mlpot_ctx is None else (
        "mini+baseline"
        if getattr(mlpot_ctx, "geometry_mini_positions", None) is not None
        and getattr(mlpot_ctx, "geometry_baseline_positions", None) is not None
        else "partial or empty"
    )
    raise RuntimeError(
        "extent repack: no readable reference coordinates on disk "
        f"({names}) or in memory ({mem})"
    )


def polish_after_extent_repack(
    mlpot_ctx: MlpotContext,
    config: DynamicsOverlapConfig,
    *,
    label: str,
) -> float:
    """Relax repacked geometry: hybrid FIRE, optional BFGS, bonded SD, MLpot mini."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _bonded_cfg_from_overlap_config,
        _resolve_mlpot_recovery_nstep,
        _resolve_pyCModel,
        _run_hybrid_bonded_mlpot_recovery,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        refresh_mlpot_energy_and_grms,
    )

    args: Any = getattr(mlpot_ctx, "workflow_args", None)
    verbose = bool(getattr(config.rescue, "verbose", False))
    grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=f"{label} pre-polish"))

    if args is not None and bool(getattr(args, "calculator_pre_minimize", True)):
        from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
            HybridCalculatorFireConfig,
            HybridCalculatorMinimizeConfig,
            coerce_hybrid_minimize_result,
            minimize_hybrid_calculator_before_sd,
            minimize_hybrid_calculator_fire_before_sd,
            resolve_calculator_mini_safe_grms,
        )

        fire_fmax = float(
            getattr(args, "rescue_fire_fmax", None)
            or getattr(args, "pre_min_fmax", 0.05)
            or 0.05
        )
        safe_grms = resolve_calculator_mini_safe_grms(args=args, context="extent_repack")
        fire_steps = int(getattr(args, "fire_min_steps", 200) or 200)
        fire_config = HybridCalculatorFireConfig(
            max_steps=fire_steps,
            fmax_ev_a=fire_fmax,
            fire_maxstep=float(getattr(args, "fire_min_maxstep", 0.2) or 0.2),
            verbose=verbose,
            max_start_grms_kcalmol_A=float("inf"),
            max_initial_fmax_ev_a=1000.0,
            safe_grms_kcalmol_A=safe_grms,
        )
        if verbose:
            print(
                f"{label}: hybrid calculator FIRE after repack "
                f"(start GRMS={grms:.4f} kcal/mol/Å, steps={fire_steps})",
                flush=True,
            )
        fire = coerce_hybrid_minimize_result(
            minimize_hybrid_calculator_fire_before_sd(
                mlpot_ctx,
                config=fire_config,
                context_prefix=f"{label} (FIRE)",
            )
        )
        grms = float(fire.grms)
        if fire.ran:
            grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))

        crossover = float(
            getattr(args, "geometry_packing_fire_bfgs_crossover_grms", 30.0) or 30.0
        )
        if grms > crossover:
            bfgs_config = HybridCalculatorMinimizeConfig(
                max_steps=int(getattr(args, "pre_min_steps", 200) or 200),
                fmax_ev_a=float(getattr(args, "pre_min_fmax", 0.05) or 0.05),
                bfgs_maxstep=float(getattr(args, "bfgs_maxstep", 0.05) or 0.05),
                verbose=verbose,
                quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)),
                max_start_grms_kcalmol_A=float("inf"),
                max_initial_fmax_ev_a=1000.0,
                safe_grms_kcalmol_A=safe_grms,
            )
            mini = coerce_hybrid_minimize_result(
                minimize_hybrid_calculator_before_sd(
                    mlpot_ctx,
                    bfgs_config,
                    context_prefix=f"{label} (BFGS)",
                )
            )
            grms = float(mini.grms)
            if mini.ran:
                grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))

    bonded_cfg = _bonded_cfg_from_overlap_config(config)
    if bonded_cfg.nstep_sd > 0 and bool(getattr(args, "bonded_mm_mini", True) if args else True):
        _run_hybrid_bonded_mlpot_recovery(
            mlpot_ctx,
            bonded_cfg,
            pyCModel=_resolve_pyCModel(mlpot_ctx, config),
            context=f"{label} bonded+MLpot",
            config=config,
        )
        setattr(mlpot_ctx, "_overlap_extent_polish_mlpot_sd_done", True)
        grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
    elif int(getattr(config, "mlpot_rescue_mini_nstep", 0) or 0) > 0:
        from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
            _run_mlpot_recovery_mini,
        )

        _run_mlpot_recovery_mini(
            mlpot_ctx,
            bonded_cfg,
            pyCModel=_resolve_pyCModel(mlpot_ctx, config),
            context=f"{label} MLpot mini",
            nstep=_resolve_mlpot_recovery_nstep(bonded_cfg, config),
            calculator_pre_minimize=False,
        )
        setattr(mlpot_ctx, "_overlap_extent_polish_mlpot_sd_done", True)
        grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))

    return grms
