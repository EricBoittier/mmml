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


def _composition_monomer_count(args: argparse.Namespace) -> int | None:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import parse_composition_dict

    comp = parse_composition_dict(getattr(args, "composition", None))
    if comp is None:
        n_mol = getattr(args, "n_molecules", None)
        return int(n_mol) if n_mol is not None else None
    return int(sum(comp.values()))


def apply_density_prep_resilient_defaults(args: argparse.Namespace) -> None:
    """Raise preventive mini/box knobs when ``liquid_prep`` or ``density_prep_mode=resilient``."""
    if not liquid_prep_enabled(args):
        return

    if getattr(args, "density_prep_ladder", None) is None:
        args.density_prep_ladder = True
    args.mc_density_equalize = True

    n_mol = _composition_monomer_count(args)
    if getattr(args, "box_size", None) is None:
        if (
            getattr(args, "target_density_g_cm3", None) is None
            and getattr(args, "bulk_density_fraction", None) is None
        ):
            if n_mol is not None and int(n_mol) > 50:
                args.bulk_density_fraction = 0.55
            else:
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
class PreMlpotGeometryGateResult:
    enabled: bool
    ran: bool
    reason: str
    steps_applied: list[str] = field(default_factory=list)
    worst_intermonomer_A: float | None = None
    aborted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ran": self.ran,
            "reason": self.reason,
            "steps_applied": list(self.steps_applied),
            "worst_intermonomer_A": self.worst_intermonomer_A,
            "aborted": self.aborted,
        }


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


def resolve_pre_mlpot_overlap_min_distance(args: argparse.Namespace) -> float:
    """Minimum inter-monomer distance for pre-MLpot hard abort (Å)."""
    dyn = getattr(args, "dynamics_overlap_min_distance", None)
    if dyn is not None and float(dyn) > 0.0:
        return float(dyn)
    build = getattr(args, "min_intermonomer_atom_distance", None)
    if build is not None and float(build) > 0.0:
        return float(build)
    return 0.5


def assert_pre_mlpot_intermonomer_geometry(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    min_distance_A: float,
    box_side: float | None,
    use_pbc: bool,
    context: str = "Pre-MLpot geometry gate",
) -> float:
    """Abort when inter-monomer atoms are closer than ``min_distance_A``."""
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import assert_no_intermonomer_atom_overlap

    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell = np.diag([float(box_side), float(box_side), float(box_side)]) if box_side else None
    return float(
        assert_no_intermonomer_atom_overlap(
            positions,
            offsets,
            min_distance=float(min_distance_A),
            cell=cell if use_pbc else None,
            context=context,
        )
    )


def _step_mc_density_at_fraction(
    args: argparse.Namespace,
    positions: np.ndarray,
    *,
    atoms_per_list: list[int],
    composition: dict[str, int] | None,
    box_side: float | None,
    charmm_pbc: bool,
    min_intermonomer_distance: float,
    density_fraction: float,
) -> tuple[np.ndarray, float | None]:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        resolve_mc_density_target_g_cm3,
    )

    full_target, _source = resolve_mc_density_target_g_cm3(args, composition)
    if full_target is None:
        return positions, box_side
    staged_target = float(full_target) * float(density_fraction)
    old_explicit = getattr(args, "mc_density_target_g_cm3", None)
    args.mc_density_target_g_cm3 = staged_target
    try:
        return _step_mc_density(
            args,
            positions,
            atoms_per_list=atoms_per_list,
            composition=composition,
            box_side=box_side,
            charmm_pbc=charmm_pbc,
            min_intermonomer_distance=min_intermonomer_distance,
        )
    finally:
        if old_explicit is None:
            if hasattr(args, "mc_density_target_g_cm3"):
                delattr(args, "mc_density_target_g_cm3")
        else:
            args.mc_density_target_g_cm3 = old_explicit


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
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import light_resync_mlpot_state
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

    if mlpot_ctx is not None:
        # Never call prepare_charmm_pbc with MLpot registered — crystal/IMAGE
        # rebuild + mlpot_update can segfault in libcharmm (see staged_workflow
        # pretreat handoff and sync_mlpot_pbc_cell_from_charmm docstrings).
        mlpot_ctx.cubic_box_side_A = float(box_side)
        mlpot_ctx.charmm_cubic_box_side_A = float(box_side)
        sync_mlpot_pbc_cell_from_charmm(
            mlpot_ctx.pyCModel,
            fallback_side_A=float(box_side),
            restart_path=pretreat_restart,
            verbose=not quiet,
        )
        light_resync_mlpot_state(
            mlpot_ctx,
            context="Density prep PBC sync" if not quiet else "",
            silent_charmm=True,
            verbose=not quiet,
            restart_path=pretreat_restart,
        )
        synced = sync_workflow_pbc_box_side_after_mm_pretreat(
            float(box_side),
            pretreat_restart=pretreat_restart,
            args=args,
            quiet=quiet,
        )
        return float(synced) if synced is not None else float(box_side)

    prepare_charmm_pbc(float(box_side))
    synced = sync_workflow_pbc_box_side_after_mm_pretreat(
        float(box_side),
        pretreat_restart=pretreat_restart,
        args=args,
        quiet=quiet,
    )
    return float(synced) if synced is not None else float(box_side)


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
                        fallback_side_A=box_side,
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
                HybridCalculatorFireConfig,
                HybridCalculatorMinimizeConfig,
                minimize_hybrid_calculator_before_sd,
                minimize_hybrid_calculator_fire_before_sd,
            )

            calc_steps = int(getattr(args, "pre_min_steps", 200) or 200)
            calc_fmax = float(getattr(args, "pre_min_fmax", 0.05) or 0.05)
            fire_steps = int(getattr(args, "fire_min_steps", 200) or 200)
            fire_fmax = float(getattr(args, "rescue_fire_fmax", calc_fmax) or calc_fmax)
            fire_config = HybridCalculatorFireConfig(
                max_steps=fire_steps,
                fmax_ev_a=fire_fmax,
                fire_maxstep=float(getattr(args, "fire_min_maxstep", 0.2) or 0.2),
                verbose=not quiet,
                max_start_grms_kcalmol_A=float(max_grms),
            )
            bfgs_config = HybridCalculatorMinimizeConfig(
                max_steps=calc_steps,
                fmax_ev_a=calc_fmax,
                bfgs_maxstep=float(getattr(args, "bfgs_maxstep", 0.05) or 0.05),
                verbose=not quiet,
                quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)),
                max_start_grms_kcalmol_A=float(max_grms),
            )

            for tag, runner, kwargs in (
                (
                    "calculator_bfgs",
                    minimize_hybrid_calculator_before_sd,
                    dict(config=bfgs_config),
                ),
                (
                    "calculator_fire",
                    minimize_hybrid_calculator_fire_before_sd,
                    dict(config=fire_config),
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


def run_pre_mlpot_geometry_gate(
    args: argparse.Namespace,
    *,
    positions: np.ndarray,
    atoms_per_list: list[int],
    composition: dict[str, int] | None,
    box_side: float | None,
    charmm_pbc: bool,
    n_mol: int,
    n_atoms: int,
) -> tuple[np.ndarray, float | None, PreMlpotGeometryGateResult]:
    """Preventive MM-only geometry ladder before MLpot registration."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    enabled = liquid_prep_enabled(args)
    if not enabled:
        return positions, box_side, PreMlpotGeometryGateResult(
            enabled=False,
            ran=False,
            reason="disabled",
        )

    quiet = bool(getattr(args, "quiet", False))
    min_overlap = resolve_pre_mlpot_overlap_min_distance(args)
    spacing = getattr(args, "spacing", None)
    seed = getattr(args, "seed", None)
    lattice_steps = int(getattr(args, "density_prep_lattice_abnr_steps", 0) or 0)
    if lattice_steps <= 0:
        lattice_steps = int(getattr(args, "mini_lattice_abnr_steps", 100) or 100)
    staged_fraction = float(getattr(args, "liquid_prep_staged_density_fraction", 0.70) or 0.70)

    result = PreMlpotGeometryGateResult(
        enabled=True,
        ran=True,
        reason="running",
    )
    pos = np.asarray(positions, dtype=np.float64)
    side = box_side

    try:
        worst = assert_pre_mlpot_intermonomer_geometry(
            pos,
            atoms_per_list,
            min_distance_A=min_overlap,
            box_side=side,
            use_pbc=charmm_pbc,
            context="Pre-MLpot gate (initial)",
        )
        result.worst_intermonomer_A = float(worst)
    except RuntimeError as exc:
        result.aborted = True
        result.reason = "overlap_abort"
        raise RuntimeError(
            f"{exc}\nPre-MLpot geometry gate: rebuild the box (Packmol/MC) or "
            f"increase spacing before MLpot registration."
        ) from exc

    step_label = "pre_mlpot:monomer_repack"
    try:
        new_pos = _step_monomer_repack(
            pos,
            atoms_per_list=list(atoms_per_list),
            box_side=side,
            min_distance=min_overlap,
            spacing=float(spacing) if spacing is not None else None,
            seed=int(seed) if seed is not None else None,
        )
        side = _sync_pbc_after_box_change(
            positions=new_pos,
            box_side=side,
            charmm_pbc=charmm_pbc,
            mlpot_ctx=None,
            args=args,
            quiet=quiet,
        )
        pos = new_pos
        result.steps_applied.append(step_label)
    except Exception as exc:
        if not quiet:
            print(f"Pre-MLpot gate: skip {step_label} ({exc})", flush=True)

    if charmm_pbc and composition is not None:
        for frac, tag in ((staged_fraction, "mc_density_staged"), (1.0, "mc_density_target")):
            step_label = f"pre_mlpot:{tag}"
            try:
                if frac < 1.0:
                    new_pos, new_side = _step_mc_density_at_fraction(
                        args,
                        pos,
                        atoms_per_list=list(atoms_per_list),
                        composition=composition,
                        box_side=side,
                        charmm_pbc=charmm_pbc,
                        min_intermonomer_distance=min_overlap,
                        density_fraction=frac,
                    )
                else:
                    new_pos, new_side = _step_mc_density(
                        args,
                        pos,
                        atoms_per_list=list(atoms_per_list),
                        composition=composition,
                        box_side=side,
                        charmm_pbc=charmm_pbc,
                        min_intermonomer_distance=min_overlap,
                    )
                if new_side is not None:
                    side = new_side
                side = _sync_pbc_after_box_change(
                    positions=new_pos,
                    box_side=side,
                    charmm_pbc=charmm_pbc,
                    mlpot_ctx=None,
                    args=args,
                    quiet=quiet,
                )
                pos = new_pos
                result.steps_applied.append(step_label)
            except Exception as exc:
                if not quiet:
                    print(f"Pre-MLpot gate: skip {step_label} ({exc})", flush=True)

    if charmm_pbc and lattice_steps > 0:
        for nocoords, tag in ((True, "lattice_box"), (False, "lattice_full")):
            step_label = f"pre_mlpot:{tag}"
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
                    fallback_side_A=side,
                )
                if new_side is not None:
                    side = float(new_side)
                result.steps_applied.append(step_label)
            except Exception as exc:
                if not quiet:
                    print(f"Pre-MLpot gate: skip {step_label} ({exc})", flush=True)

    try:
        worst = assert_pre_mlpot_intermonomer_geometry(
            pos,
            atoms_per_list,
            min_distance_A=min_overlap,
            box_side=side,
            use_pbc=charmm_pbc,
            context="Pre-MLpot gate (final)",
        )
        result.worst_intermonomer_A = float(worst)
    except RuntimeError as exc:
        result.aborted = True
        result.reason = "overlap_abort_post_ladder"
        raise RuntimeError(
            f"{exc}\nPre-MLpot geometry gate: overlap persists after preventive ladder."
        ) from exc

    sync_charmm_positions(pos)
    result.reason = "ok"
    if not quiet:
        print(
            f"Pre-MLpot geometry gate: {len(result.steps_applied)} step(s), "
            f"worst inter-monomer distance {result.worst_intermonomer_A:.3f} Å",
            flush=True,
        )
    return pos, side, result


def run_geometry_packing_recovery(
    mlpot_ctx: Any,
    *,
    args: argparse.Namespace,
    atoms_per_list: list[int],
    composition: dict[str, int] | None,
    box_side: float | None,
    charmm_pbc: bool,
    context_prefix: str = "Geometry packing recovery",
    calculator_minimize: bool = True,
    calculator_minimize_steps: int = 200,
    calculator_minimize_fmax_ev_a: float = 0.05,
    calculator_bfgs_maxstep: float = 0.05,
    calculator_fire_steps: int = 200,
    calculator_fire_fmax_ev_a: float | None = None,
    calculator_fire_maxstep: float = 0.2,
    quiet_bfgs: bool = False,
    verbose: bool = True,
    grms_limit: float | None = None,
) -> float:
    """Repack / MC / FIRE / BFGS path for ``geometry_stress`` (skip bonded-MM first)."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        refresh_mlpot_energy_and_grms,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    quiet = not verbose
    min_overlap = resolve_pre_mlpot_overlap_min_distance(args)
    spacing = getattr(args, "spacing", None)
    seed = getattr(args, "seed", None)
    grms = refresh_mlpot_energy_and_grms(
        mlpot_ctx,
        context=f"{context_prefix} (initial)" if verbose else "",
    )

    step_label = f"{context_prefix}:monomer_repack"
    try:
        pos = get_charmm_positions_array()
        new_pos = _step_monomer_repack(
            pos,
            atoms_per_list=list(atoms_per_list),
            box_side=box_side,
            min_distance=min_overlap,
            spacing=float(spacing) if spacing is not None else None,
            seed=int(seed) if seed is not None else None,
        )
        box_side = _sync_pbc_after_box_change(
            positions=new_pos,
            box_side=box_side,
            charmm_pbc=charmm_pbc,
            mlpot_ctx=mlpot_ctx,
            args=args,
            quiet=quiet,
        )
        grms = refresh_mlpot_energy_and_grms(
            mlpot_ctx,
            context=f"{context_prefix} ({step_label})" if verbose else "",
        )
    except Exception as exc:
        if verbose:
            print(f"{context_prefix}: skip monomer_repack ({exc})", flush=True)

    if charmm_pbc and composition is not None:
        step_label = f"{context_prefix}:mc_density"
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
            grms = refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"{context_prefix} ({step_label})" if verbose else "",
            )
        except Exception as exc:
            if verbose:
                print(f"{context_prefix}: skip mc_density ({exc})", flush=True)

    if calculator_minimize:
        from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
            HybridCalculatorFireConfig,
            HybridCalculatorMinimizeConfig,
            minimize_hybrid_calculator_before_sd,
            minimize_hybrid_calculator_fire_before_sd,
        )

        fire_fmax = (
            float(calculator_fire_fmax_ev_a)
            if calculator_fire_fmax_ev_a is not None
            else float(calculator_minimize_fmax_ev_a)
        )
        start_cap = float(grms_limit) if grms_limit is not None else float("inf")
        fire_bfgs_crossover = float(
            getattr(args, "geometry_packing_fire_bfgs_crossover_grms", 30.0) or 30.0
        )
        bfgs_first = bool(np.isfinite(grms) and float(grms) > fire_bfgs_crossover)
        fire_config = HybridCalculatorFireConfig(
            max_steps=int(calculator_fire_steps),
            fmax_ev_a=fire_fmax,
            fire_maxstep=float(calculator_fire_maxstep),
            verbose=verbose,
            max_start_grms_kcalmol_A=start_cap,
        )
        bfgs_config = HybridCalculatorMinimizeConfig(
            max_steps=int(calculator_minimize_steps),
            fmax_ev_a=float(calculator_minimize_fmax_ev_a),
            bfgs_maxstep=float(calculator_bfgs_maxstep),
            verbose=verbose,
            quiet_bfgs=quiet_bfgs,
            max_start_grms_kcalmol_A=start_cap,
            max_initial_fmax_ev_a=1000.0,
        )

        def _run_bfgs() -> float:
            nonlocal grms
            grms = minimize_hybrid_calculator_before_sd(
                mlpot_ctx,
                bfgs_config,
                context_prefix=f"{context_prefix} (BFGS)",
            )
            return refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"{context_prefix} (post-BFGS)" if verbose else "",
            )

        def _run_fire() -> float:
            nonlocal grms
            grms = minimize_hybrid_calculator_fire_before_sd(
                mlpot_ctx,
                config=fire_config,
                context_prefix=f"{context_prefix} (FIRE)",
            )
            return refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context=f"{context_prefix} (post-FIRE)" if verbose else "",
            )

        if verbose and bfgs_first:
            print(
                f"{context_prefix}: GRMS {grms:.1f} > {fire_bfgs_crossover:.1f}; "
                "running guarded BFGS before FIRE",
                flush=True,
            )

        runners = (_run_bfgs, _run_fire) if bfgs_first else (_run_fire, _run_bfgs)
        for runner in runners:
            try:
                runner()
            except Exception as exc:
                if verbose:
                    print(f"{context_prefix}: calculator mini skipped ({exc})", flush=True)

    return float(grms)
