"""Resilient density/box preparation ladder for condensed-phase md-system runs.

When Packmol builds a dense liquid box at target density, minimization can stall
with high hybrid GRMS.  This module applies preventive defaults (``resilient``
mode) and an optional post-mini rescue ladder that chains geometry, box-sizing,
CHARMM, ASE, and MLpot recovery steps until GRMS is low enough for dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

DensityPrepMode = Literal["off", "resilient"]


def resolve_density_prep_mode(args: argparse.Namespace) -> DensityPrepMode:
    raw = getattr(args, "density_prep_mode", None)
    if raw is None:
        return "off"
    mode = str(raw).strip().lower()
    if mode in ("off", "resilient"):
        return mode  # type: ignore[return-value]
    raise ValueError(f"--density-prep-mode must be 'off' or 'resilient', got {raw!r}")


def liquid_prep_enabled(args: argparse.Namespace) -> bool:
    """True when the dense-liquid prep stack should run (shorthand or explicit mode)."""
    if bool(getattr(args, "liquid_prep", False)):
        return True
    return resolve_density_prep_mode(args) == "resilient"


def density_prep_ladder_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "density_prep_ladder", None)
    if explicit is not None:
        return bool(explicit)
    return liquid_prep_enabled(args)


def _bump_int_attr(args: argparse.Namespace, name: str, floor: int) -> None:
    current = int(getattr(args, name, 0) or 0)
    if current < floor:
        setattr(args, name, floor)


def apply_density_prep_resilient_defaults(args: argparse.Namespace) -> None:
    """Raise preventive mini/box knobs when ``liquid_prep`` or ``density_prep_mode=resilient``."""
    if not liquid_prep_enabled(args):
        return

    if getattr(args, "density_prep_ladder", None) is None:
        args.density_prep_ladder = True
    args.mc_density_equalize = True

    if getattr(args, "box_size", None) is None:
        if (
            getattr(args, "target_density_g_cm3", None) is None
            and getattr(args, "bulk_density_fraction", None) is None
        ):
            args.bulk_density_fraction = 0.75

    _bump_int_attr(args, "charmm_sd_steps", 1000)
    _bump_int_attr(args, "charmm_abnr_steps", 1000)
    _bump_int_attr(args, "mini_nstep", 500)
    _bump_int_attr(args, "bonded_mm_mini_steps", 500)
    if int(getattr(args, "mini_lattice_abnr_steps", 0) or 0) <= 0:
        args.mini_lattice_abnr_steps = 200
    if float(getattr(args, "mini_box_equil_ps", 0.0) or 0.0) <= 0.0:
        args.mini_box_equil_ps = 2.0

    if getattr(args, "box_size", None) is not None:
        args.mini_lattice_abnr_allow_fixed_box = True
        args.mini_box_equil_allow_fixed_box = True

    args.calculator_pre_minimize = bool(getattr(args, "calculator_pre_minimize", True))


@dataclass
class DensityPrepLadderResult:
    enabled: bool
    ran: bool
    reason: str
    initial_grms: float | None = None
    final_grms: float | None = None
    rounds: int = 0
    steps_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ran": self.ran,
            "reason": self.reason,
            "initial_grms": self.initial_grms,
            "final_grms": self.final_grms,
            "rounds": self.rounds,
            "steps_applied": list(self.steps_applied),
        }


def _step_monomer_repack(
    positions: np.ndarray,
    *,
    atoms_per_list: list[int],
    box_side: float | None,
    min_distance: float,
    spacing: float | None,
    seed: int | None,
) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import repack_monomers_clear_overlap

    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell = np.diag([float(box_side), float(box_side), float(box_side)]) if box_side else None
    return repack_monomers_clear_overlap(
        positions,
        offsets,
        min_distance=float(min_distance),
        spacing=spacing,
        seed=seed,
        cell=cell,
    )


def _step_mc_density(
    args: argparse.Namespace,
    positions: np.ndarray,
    *,
    atoms_per_list: list[int],
    composition: dict[str, int] | None,
    box_side: float | None,
    charmm_pbc: bool,
    min_intermonomer_distance: float,
) -> tuple[np.ndarray, float | None]:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )

    new_pos, new_side, summary = apply_mc_density_equalization(
        args,
        positions,
        atoms_per_list=atoms_per_list,
        composition=composition,
        box_side_A=box_side,
        use_pbc=charmm_pbc,
        handoff_present=True,
        min_intermonomer_distance_A=float(min_intermonomer_distance),
    )
    if summary.ran and new_side is not None:
        return new_pos, float(new_side)
    return positions, box_side


def _sync_pbc_after_box_change(
    *,
    positions: np.ndarray,
    box_side: float | None,
    charmm_pbc: bool,
    mlpot_ctx: Any,
    pretreat_restart: Path | None = None,
    args: argparse.Namespace | None = None,
    quiet: bool = False,
) -> float | None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        prepare_charmm_pbc,
        sync_workflow_pbc_box_side_after_mm_pretreat,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        sync_mlpot_pbc_cell_from_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(positions)
    if not charmm_pbc or box_side is None:
        return box_side
    prepare_charmm_pbc(float(box_side))
    synced = sync_workflow_pbc_box_side_after_mm_pretreat(
        float(box_side),
        pretreat_restart=pretreat_restart,
        args=args,
        quiet=quiet,
    )
    if synced is not None:
        box_side = float(synced)
        mlpot_ctx.cubic_box_side_A = box_side
        mlpot_ctx.charmm_cubic_box_side_A = box_side
        sync_mlpot_pbc_cell_from_charmm(
            mlpot_ctx.pyCModel,
            fallback_side_A=box_side,
            restart_path=None,
            verbose=not quiet,
        )
    return box_side


def run_density_prep_ladder(
    args: argparse.Namespace,
    *,
    mlpot_ctx: Any,
    pyCModel: Any,
    max_grms: float,
    current_grms: float,
    n_mol: int,
    n_atoms: int,
    box_side: float | None,
    charmm_pbc: bool,
    atoms_per_list: list[int] | None,
    composition: dict[str, int] | None,
    mini_nstep: int,
    mini_nprint: int,
    fix_resids: list[int],
    show_energy: bool,
    z: np.ndarray,
) -> tuple[float, float | None, DensityPrepLadderResult]:
    """Attempt staged recovery until hybrid GRMS is below ``max_grms``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        refresh_mlpot_energy_and_grms,
        resolve_test_first_config,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    enabled = density_prep_ladder_enabled(args)
    if not enabled:
        return current_grms, box_side, DensityPrepLadderResult(
            enabled=False,
            ran=False,
            reason="disabled",
            initial_grms=current_grms,
            final_grms=current_grms,
        )
    if current_grms <= max_grms:
        return current_grms, box_side, DensityPrepLadderResult(
            enabled=True,
            ran=False,
            reason="grms_ok",
            initial_grms=current_grms,
            final_grms=current_grms,
        )
    if atoms_per_list is None:
        return current_grms, box_side, DensityPrepLadderResult(
            enabled=True,
            ran=False,
            reason="missing_atoms_per_list",
            initial_grms=current_grms,
            final_grms=current_grms,
        )

    max_rounds = max(1, int(getattr(args, "density_prep_ladder_max_rounds", 3) or 3))
    min_overlap = float(getattr(args, "min_intermonomer_atom_distance", 0.1) or 0.1)
    spacing = getattr(args, "spacing", None)
    seed = getattr(args, "seed", None)
    lattice_steps = int(getattr(args, "density_prep_lattice_abnr_steps", 0) or 0)
    if lattice_steps <= 0:
        lattice_steps = int(getattr(args, "mini_lattice_abnr_steps", 100) or 100)
    bonded_steps = int(getattr(args, "bonded_mm_mini_steps", 200) or 200)
    quiet = bool(getattr(args, "quiet", False))

    result = DensityPrepLadderResult(
        enabled=True,
        ran=True,
        reason="running",
        initial_grms=current_grms,
        final_grms=current_grms,
    )
    grms = float(current_grms)

    if not quiet:
        print(
            f"\nDensity prep ladder: GRMS {grms:.4f} > {max_grms:.4f}; "
            f"up to {max_rounds} round(s)",
            flush=True,
        )

    for round_idx in range(max_rounds):
        if grms <= max_grms:
            break
        result.rounds = round_idx + 1
        pos = get_charmm_positions_array()

        step_label = f"round{round_idx + 1}:monomer_repack"
        try:
            new_pos = _step_monomer_repack(
                pos,
                atoms_per_list=list(atoms_per_list),
                box_side=box_side,
                min_distance=min_overlap,
                spacing=float(spacing) if spacing is not None else None,
                seed=int(seed) + round_idx if seed is not None else None,
            )
            box_side = _sync_pbc_after_box_change(
                positions=new_pos,
                box_side=box_side,
                charmm_pbc=charmm_pbc,
                mlpot_ctx=mlpot_ctx,
                args=args,
                quiet=quiet,
            )
            result.steps_applied.append(step_label)
            grms = refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"Density prep ladder ({step_label})" if not quiet else "",
            )
        except Exception as exc:
            if not quiet:
                print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)

        if grms <= max_grms:
            break

        if charmm_pbc and composition is not None:
            step_label = f"round{round_idx + 1}:mc_density"
            try:
                pos = get_charmm_positions_array()
                new_pos, new_side = _step_mc_density(
                    args,
                    pos,
                    atoms_per_list=list(atoms_per_list),
                    composition=composition,
                    box_side=box_side,
                    charmm_pbc=charmm_pbc,
                    min_intermonomer_distance=min_overlap,
                )
                if new_side is not None:
                    box_side = new_side
                box_side = _sync_pbc_after_box_change(
                    positions=new_pos,
                    box_side=box_side,
                    charmm_pbc=charmm_pbc,
                    mlpot_ctx=mlpot_ctx,
                    args=args,
                    quiet=quiet,
                )
                result.steps_applied.append(step_label)
                grms = refresh_mlpot_energy_and_grms(
                    mlpot_ctx,
                    context=f"Density prep ladder ({step_label})" if not quiet else "",
                )
            except Exception as exc:
                if not quiet:
                    print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)

        if grms <= max_grms:
            break

        if charmm_pbc and lattice_steps > 0:
            for nocoords, tag in ((True, "lattice_box"), (False, "lattice_full")):
                step_label = f"round{round_idx + 1}:{tag}"
                try:
                    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
                        run_charmm_lattice_abnr,
                    )

                    new_side = run_charmm_lattice_abnr(
                        nstep=lattice_steps,
                        tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                        tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                        nocoords=nocoords,
                        verbose=not quiet,
                    )
                    if new_side is not None:
                        box_side = float(new_side)
                        mlpot_ctx.cubic_box_side_A = box_side
                        mlpot_ctx.charmm_cubic_box_side_A = box_side
                    result.steps_applied.append(step_label)
                    grms = refresh_mlpot_energy_and_grms(
                        mlpot_ctx,
                        context=f"Density prep ladder ({step_label})" if not quiet else "",
                    )
                except Exception as exc:
                    if not quiet:
                        print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)
                if grms <= max_grms:
                    break

        if grms <= max_grms:
            break

        step_label = f"round{round_idx + 1}:bonded_mm"
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                BondedMmMiniConfig,
                minimize_bonded_mm_recovery,
            )

            minimize_bonded_mm_recovery(
                mlpot_ctx,
                BondedMmMiniConfig(
                    nstep_sd=bonded_steps,
                    nprint=max(1, mini_nprint),
                    tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                    tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                    verbose=not quiet,
                    show_energy=False,
                ),
            )
            result.steps_applied.append(step_label)
            grms = refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"Density prep ladder ({step_label})" if not quiet else "",
            )
        except Exception as exc:
            if not quiet:
                print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)

        if grms <= max_grms:
            break

        if bool(getattr(args, "calculator_pre_minimize", True)):
            from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
                HybridCalculatorMinimizeConfig,
                minimize_hybrid_calculator_before_sd,
                minimize_hybrid_calculator_fire_before_sd,
            )

            calc_steps = int(getattr(args, "pre_min_steps", 200) or 200)
            calc_fmax = float(getattr(args, "pre_min_fmax", 0.05) or 0.05)
            fire_steps = int(getattr(args, "fire_min_steps", 200) or 200)
            fire_fmax = float(getattr(args, "rescue_fire_fmax", calc_fmax) or calc_fmax)

            for tag, runner, kwargs in (
                (
                    "calculator_bfgs",
                    minimize_hybrid_calculator_before_sd,
                    dict(
                        config=HybridCalculatorMinimizeConfig(
                            max_steps=calc_steps,
                            fmax_ev_a=calc_fmax,
                            bfgs_maxstep=float(getattr(args, "bfgs_maxstep", 0.05) or 0.05),
                            verbose=not quiet,
                            quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)),
                            max_start_grms_kcalmol_A=float(max_grms),
                        ),
                    ),
                ),
                (
                    "calculator_fire",
                    minimize_hybrid_calculator_fire_before_sd,
                    dict(
                        max_steps=fire_steps,
                        fmax_ev_a=fire_fmax,
                        fire_maxstep=float(getattr(args, "fire_min_maxstep", 0.2) or 0.2),
                        verbose=not quiet,
                        max_start_grms_kcalmol_A=float(max_grms),
                    ),
                ),
            ):
                step_label = f"round{round_idx + 1}:{tag}"
                try:
                    grms = runner(
                        mlpot_ctx,
                        context_prefix=f"Density prep ladder ({step_label})",
                        **kwargs,
                    )
                    result.steps_applied.append(step_label)
                    grms = refresh_mlpot_energy_and_grms(
                        mlpot_ctx,
                        context=f"Density prep ladder ({step_label})" if not quiet else "",
                    )
                except Exception as exc:
                    if not quiet:
                        print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)
                if grms <= max_grms:
                    break

        if grms <= max_grms:
            break

        step_label = f"round{round_idx + 1}:mlpot_sd"
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                MinimizeWithMlpotConfig,
                minimize_with_mlpot,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_resids

            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            minimize_with_mlpot(
                MinimizeWithMlpotConfig(
                    fixed_ml_selection=fix_sel,
                    nstep=int(mini_nstep),
                    nprint=int(mini_nprint),
                    verbose=not quiet,
                    reference_positions=get_charmm_positions_array(),
                    pyCModel=pyCModel,
                    mlpot_ctx=mlpot_ctx,
                    save=False,
                    title=f"Density prep ladder {step_label}",
                    skip_if_crd_exists=False,
                    test_first=resolve_test_first_config(args),
                    show_energy=show_energy,
                    pre_sd_bonded_recovery_grms_kcalmol_A=float(max_grms),
                    pre_sd_bonded_recovery_nstep=bonded_steps,
                    calculator_pre_minimize=False,
                )
            )
            result.steps_applied.append(step_label)
            grms = refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"Density prep ladder ({step_label})" if not quiet else "",
            )
        except Exception as exc:
            if not quiet:
                print(f"Density prep ladder: skip {step_label} ({exc})", flush=True)

    result.final_grms = float(grms)
    result.reason = "grms_ok" if grms <= max_grms else "grms_still_high"
    if not quiet:
        print(
            f"Density prep ladder done: GRMS {result.initial_grms:.4f} -> "
            f"{result.final_grms:.4f} (limit {max_grms:.4f}); "
            f"rounds={result.rounds}, steps={len(result.steps_applied)}",
            flush=True,
        )
    return float(grms), box_side, result
