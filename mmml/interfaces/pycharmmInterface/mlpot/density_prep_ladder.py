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

_DEFAULT_GEOMETRY_PREP_REGRESS_RATIO = 1.25
_DEFAULT_GEOMETRY_PREP_REGRESS_MIN_DELTA = 50.0
# Lattice ABNR rebuilds IMAGE/MAKINB lists; unsafe on catastrophic overlaps.
_LATTICE_ABNR_GRMS_STRESS_CEILING = 500.0


def _geometry_prep_regressed(
    grms_before: float,
    grms_after: float,
    *,
    ratio: float = _DEFAULT_GEOMETRY_PREP_REGRESS_RATIO,
    min_delta: float = _DEFAULT_GEOMETRY_PREP_REGRESS_MIN_DELTA,
) -> bool:
    """True when a prep step materially worsened hybrid GRMS."""
    if not np.isfinite(grms_before) or not np.isfinite(grms_after):
        return False
    before = float(grms_before)
    after = float(grms_after)
    return after > max(before * float(ratio), before + float(min_delta))


def _rollback_charmm_geometry(
    positions: np.ndarray,
    mlpot_ctx: Any,
    *,
    quiet: bool = True,
) -> float:
    """Restore CHARMM coordinates and refresh hybrid GRMS after a failed prep step."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import refresh_mlpot_energy_and_grms
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(np.asarray(positions, dtype=np.float64))
    return float(
        refresh_mlpot_energy_and_grms(
            mlpot_ctx,
            context="",
            verbose=not quiet,
        )
    )


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
    skip_box_prep = certified_box_handoff(args)
    if skip_box_prep:
        args.mini_lattice_abnr_steps = 0
        args.density_prep_lattice_abnr_steps = 0
        args.mini_box_equil_ps = 0.0
    elif int(getattr(args, "mini_lattice_abnr_steps", 0) or 0) <= 0:
        args.mini_lattice_abnr_steps = 200
        if float(getattr(args, "mini_box_equil_ps", 0.0) or 0.0) <= 0.0:
            args.mini_box_equil_ps = 2.0

    if getattr(args, "min_intermonomer_atom_distance", None) is None:
        from mmml.utils.intermonomer_geometry import DEFAULT_PRE_MLPOT_OVERLAP_MIN_A

        args.min_intermonomer_atom_distance = float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)

    if getattr(args, "box_size", None) is not None:
        args.mini_lattice_abnr_allow_fixed_box = True
        args.mini_box_equil_allow_fixed_box = True

    args.calculator_pre_minimize = bool(getattr(args, "calculator_pre_minimize", True))
    _bump_int_attr(args, "pre_min_steps", 200)
    _bump_int_attr(args, "fire_min_steps", 200)


def condensed_phase_md_prep_recommended(args: argparse.Namespace) -> bool:
    """True when md-system should use dense-liquid prep defaults without Packmol."""
    if liquid_prep_enabled(args):
        return True
    if getattr(args, "from_psf", None) or getattr(args, "skip_cluster_build", False):
        return True
    n_mol = _composition_monomer_count(args)
    return n_mol is not None and int(n_mol) >= 15


def certified_box_handoff(args: argparse.Namespace) -> bool:
    """True when md-system loads a pre-built liquid-box artifact (skip Packmol rebuild)."""
    return bool(getattr(args, "skip_cluster_build", False)) and bool(
        getattr(args, "from_psf", None) or getattr(args, "from_crd", None)
    )


def resolve_density_prep_lattice_abnr_steps(args: argparse.Namespace) -> int:
    """Lattice ABNR steps for pre-MLpot gate / density-prep ladder (0 = skip).

    ``mini_lattice_abnr_steps=0`` must stay disabled — do not fall back to 100.
    Certified liquid-box handoffs skip lattice work (box already optimized there).
    """
    if certified_box_handoff(args):
        explicit = int(getattr(args, "density_prep_lattice_abnr_steps", 0) or 0)
        if explicit > 0:
            return explicit
        mini = int(getattr(args, "mini_lattice_abnr_steps", 0) or 0)
        if mini > 0:
            return mini
        return 0

    steps = int(getattr(args, "density_prep_lattice_abnr_steps", 0) or 0)
    if steps > 0:
        return steps
    steps = int(getattr(args, "mini_lattice_abnr_steps", 0) or 0)
    if steps > 0:
        return steps
    return 100


def apply_condensed_phase_md_defaults(args: argparse.Namespace) -> None:
    """Apply resilient liquid prep defaults for certified-box / large-cluster md-system runs."""
    if not condensed_phase_md_prep_recommended(args):
        return
    if not liquid_prep_enabled(args):
        args.liquid_prep = True
    apply_density_prep_resilient_defaults(args)


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
    mlpot_ctx: Any | None = None,
    max_select: int = 2,
    verbose: bool = False,
) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import (
        repack_monomers_clear_overlap,
        repack_selected_monomers_clear_overlap,
    )

    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell = np.diag([float(box_side), float(box_side), float(box_side)]) if box_side else None

    if mlpot_ctx is not None:
        from mmml.utils.monomer_force_diag import resolve_selective_repack_monomers

        diag = resolve_selective_repack_monomers(
            mlpot_ctx,
            offsets,
            max_select=max_select,
            positions=positions,
        )
        if diag is not None:
            if verbose:
                flagged_txt = ", ".join(str(i) for i in diag.flagged)
                grms_txt = ", ".join(
                    f"{i}:{diag.grms_per_monomer[i]:.1f}" for i in diag.flagged
                )
                print(
                    f"Selective monomer repack: patch [{flagged_txt}] "
                    f"(per-mono GRMS {grms_txt} kcal/mol/Å; "
                    f"cluster {diag.cluster_grms:.1f})",
                    flush=True,
                )
            return repack_selected_monomers_clear_overlap(
                positions,
                offsets,
                list(diag.flagged),
                min_distance=float(min_distance),
                spacing=spacing,
                margin=0.2,
                seed=seed,
                cell=cell,
            )

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
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance as _resolve

    return _resolve(args)


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
    report_resync: bool = True,
) -> float | None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import light_resync_mlpot_state
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        push_charmm_cubic_box_side_A,
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
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                charmm_crystal_is_active,
                sync_charmm_crystal_after_mm_pretreat,
            )

            if not charmm_crystal_is_active():
                sync_charmm_crystal_after_mm_pretreat(float(box_side), quiet=quiet)
        except Exception:
            pass
        sync_mlpot_pbc_cell_from_charmm(
            mlpot_ctx.pyCModel,
            fallback_side_A=float(box_side),
            restart_path=pretreat_restart,
            verbose=not quiet,
        )
        light_resync_mlpot_state(
            mlpot_ctx,
            context="Density prep PBC sync" if (not quiet and report_resync) else "",
            silent_charmm=True,
            verbose=not quiet and report_resync,
            restart_path=pretreat_restart,
        )
        synced = sync_workflow_pbc_box_side_after_mm_pretreat(
            float(box_side),
            pretreat_restart=pretreat_restart,
            args=args,
            quiet=quiet,
        )
        return float(synced) if synced is not None else float(box_side)

    pushed, _ = push_charmm_cubic_box_side_A(
        float(box_side),
        quiet=quiet,
    )
    synced = sync_workflow_pbc_box_side_after_mm_pretreat(
        float(pushed),
        pretreat_restart=pretreat_restart,
        args=args,
        quiet=quiet,
    )
    return float(synced) if synced is not None else float(pushed)


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
    min_overlap = resolve_pre_mlpot_overlap_min_distance(args)
    spacing = getattr(args, "spacing", None)
    seed = getattr(args, "seed", None)
    lattice_steps = resolve_density_prep_lattice_abnr_steps(args)
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

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import measure_hybrid_charmm_grms
    from mmml.utils.prep_ladder_report import PrepLadderJournal, PrepMetrics

    journal = PrepLadderJournal(quiet=quiet)
    journal.begin(initial_grms=grms, max_grms=float(max_grms), max_rounds=max_rounds)
    from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
        RecoveryProgressStore,
    )

    progress = RecoveryProgressStore.for_prep_ladder(args, quiet=quiet)
    if progress is not None:
        progress.record_step("initial", grms_kcalmol_A=grms, box_side_A=box_side)

    def _step_metrics(step_grms: float) -> PrepMetrics:
        diag = measure_hybrid_charmm_grms(mlpot_ctx)
        return PrepMetrics.from_mlpot(
            mlpot_ctx,
            hybrid_grms=float(step_grms),
            charmm_grms=float(diag.charmm),
            diag_kind=str(diag.kind),
        )

    def _refresh_after_step(step_label: str) -> float:
        step_grms = refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
        journal.record_step(step_label, _step_metrics(step_grms))
        if progress is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                get_charmm_positions_array,
            )

            progress.record_step(
                step_label,
                grms_kcalmol_A=float(step_grms),
                box_side_A=box_side,
                positions=get_charmm_positions_array(),
            )
        return float(step_grms)

    for round_idx in range(max_rounds):
        if grms <= max_grms:
            break
        result.rounds = round_idx + 1
        journal.begin_round(round_idx, grms)
        round_grms_start = float(grms)
        round_pos_start = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
        pos = round_pos_start

        step_label = f"round{round_idx + 1}:monomer_repack"
        try:
            pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64, copy=True)
            grms_before = float(grms)
            new_pos = _step_monomer_repack(
                pos_before,
                atoms_per_list=list(atoms_per_list),
                box_side=box_side,
                min_distance=min_overlap,
                spacing=float(spacing) if spacing is not None else None,
                seed=int(seed) + round_idx if seed is not None else None,
                mlpot_ctx=mlpot_ctx,
                verbose=not quiet,
            )
            box_side = _sync_pbc_after_box_change(
                positions=new_pos,
                box_side=box_side,
                charmm_pbc=charmm_pbc,
                mlpot_ctx=mlpot_ctx,
                args=args,
                quiet=quiet,
                report_resync=False,
            )
            grms_after = float(_refresh_after_step(step_label))
            if _geometry_prep_regressed(grms_before, grms_after):
                if not quiet:
                    print(
                        f"{step_label}: rollback monomer_repack "
                        f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                        flush=True,
                    )
                grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
                journal.skip_step(
                    step_label,
                    f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
                )
            else:
                grms = grms_after
                result.steps_applied.append(step_label)
        except Exception as exc:
            journal.skip_step(step_label, str(exc))

        if grms <= max_grms:
            break

        if charmm_pbc and composition is not None:
            step_label = f"round{round_idx + 1}:mc_density"
            try:
                pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64, copy=True)
                grms_before = float(grms)
                new_pos, new_side = _step_mc_density(
                    args,
                    pos_before,
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
                    report_resync=False,
                )
                grms_after = float(_refresh_after_step(step_label))
                if _geometry_prep_regressed(grms_before, grms_after):
                    if not quiet:
                        print(
                            f"{step_label}: rollback mc_density "
                            f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                            flush=True,
                        )
                    grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
                    journal.skip_step(
                        step_label,
                        f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
                    )
                else:
                    grms = grms_after
                    result.steps_applied.append(step_label)
            except Exception as exc:
                journal.skip_step(step_label, str(exc))

        if grms <= max_grms:
            break

        if charmm_pbc and lattice_steps > 0:
            if float(grms) > _LATTICE_ABNR_GRMS_STRESS_CEILING:
                journal.skip_step(
                    f"round{round_idx + 1}:lattice_skipped",
                    (
                        f"hybrid GRMS {float(grms):.1f} > "
                        f"{_LATTICE_ABNR_GRMS_STRESS_CEILING:.0f} kcal/mol/Å "
                        "(lattice ABNR unsafe on geometry_stress)"
                    ),
                )
            else:
                for nocoords, tag in ((False, "lattice_full"), (True, "lattice_box")):
                    step_label = f"round{round_idx + 1}:{tag}"
                    try:
                        pos_before = np.asarray(
                            get_charmm_positions_array(), dtype=np.float64
                        ).copy()
                        grms_before = float(grms)
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
                            allow_prepare_pbc=mlpot_ctx is None,
                        )
                        if new_side is not None:
                            box_side = float(new_side)
                            mlpot_ctx.cubic_box_side_A = box_side
                            mlpot_ctx.charmm_cubic_box_side_A = box_side
                        grms_after = float(_refresh_after_step(step_label))
                        if _geometry_prep_regressed(grms_before, grms_after):
                            if not quiet:
                                print(
                                    f"{step_label}: rollback {tag} "
                                    f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                                    flush=True,
                                )
                            grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
                            journal.skip_step(
                                step_label,
                                f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
                            )
                        else:
                            grms = grms_after
                            result.steps_applied.append(step_label)
                    except Exception as exc:
                        journal.skip_step(step_label, str(exc))
                    if grms <= max_grms:
                        break

        if grms <= max_grms:
            break

        step_label = f"round{round_idx + 1}:bonded_mm"
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                _mlpot_covers_all_atoms,
                _run_mlpot_recovery_mini,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                bonded_mm_mini_config_from_namespace,
                minimize_bonded_mm_recovery,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics import (
                print_geometry_checkpoint_diff,
            )

            pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64, copy=True)
            grms_before = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
            bonded_cfg = bonded_mm_mini_config_from_namespace(
                args,
                nstep_sd=bonded_steps,
                tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                verbose=not quiet,
                show_energy=False,
            )
            if _mlpot_covers_all_atoms(mlpot_ctx):
                if not quiet:
                    print(
                        f"{step_label}: all-ML cluster — skipping CHARMM bonded-MM SD "
                        "(CGENFF bonded on ML atoms distorts PhysNet geometry); "
                        "using MLpot SD recovery instead",
                        flush=True,
                    )
                _run_mlpot_recovery_mini(
                    mlpot_ctx,
                    bonded_cfg,
                    pyCModel=pyCModel,
                    context=f"Density prep ladder ({step_label})",
                    nstep=bonded_steps,
                    calculator_pre_minimize=False,
                )
            else:
                minimize_bonded_mm_recovery(
                    mlpot_ctx,
                    bonded_cfg,
                    topology_psf=getattr(mlpot_ctx, "topology_psf_path", None),
                )
            print_geometry_checkpoint_diff(
                pos_before,
                get_charmm_positions_array(),
                step_label=step_label,
                mlpot_ctx=mlpot_ctx,
            )
            grms_after = float(_refresh_after_step(step_label))
            if _geometry_prep_regressed(grms_before, grms_after):
                if not quiet:
                    print(
                        f"{step_label}: rollback bonded_mm "
                        f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                        flush=True,
                    )
                grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
                journal.skip_step(
                    step_label,
                    f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
                )
            else:
                grms = grms_after
                result.steps_applied.append(step_label)
        except Exception as exc:
            journal.skip_step(step_label, str(exc))

        if grms <= max_grms:
            break

        if bool(getattr(args, "calculator_pre_minimize", True)):
            from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
                HybridCalculatorFireConfig,
                HybridCalculatorMinimizeConfig,
                coerce_hybrid_minimize_result,
                minimize_hybrid_calculator_before_sd,
                minimize_hybrid_calculator_fire_before_sd,
                resolve_calculator_mini_safe_grms,
            )

            calc_steps = int(getattr(args, "pre_min_steps", 200) or 200)
            calc_fmax = float(getattr(args, "pre_min_fmax", 0.05) or 0.05)
            fire_steps = int(getattr(args, "fire_min_steps", 200) or 200)
            fire_fmax = float(getattr(args, "rescue_fire_fmax", calc_fmax) or calc_fmax)
            safe_grms = resolve_calculator_mini_safe_grms(
                args=args,
                context="density_prep",
            )
            fire_config = HybridCalculatorFireConfig(
                max_steps=fire_steps,
                fmax_ev_a=fire_fmax,
                fire_maxstep=float(getattr(args, "fire_min_maxstep", 0.2) or 0.2),
                verbose=not quiet,
                max_start_grms_kcalmol_A=float(max_grms),
                safe_grms_kcalmol_A=safe_grms,
            )
            bfgs_config = HybridCalculatorMinimizeConfig(
                max_steps=calc_steps,
                fmax_ev_a=calc_fmax,
                bfgs_maxstep=float(getattr(args, "bfgs_maxstep", 0.05) or 0.05),
                verbose=not quiet,
                quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)),
                max_start_grms_kcalmol_A=float(max_grms),
                safe_grms_kcalmol_A=safe_grms,
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
                    mini = coerce_hybrid_minimize_result(
                        runner(
                            mlpot_ctx,
                            context_prefix=f"Density prep ladder ({step_label})",
                            **kwargs,
                        )
                    )
                    grms = float(mini.grms)
                    if mini.ran:
                        result.steps_applied.append(step_label)
                    grms = _refresh_after_step(step_label)
                except Exception as exc:
                    journal.skip_step(step_label, str(exc))
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
            grms = _refresh_after_step(step_label)
        except Exception as exc:
            journal.skip_step(step_label, str(exc))

        if _geometry_prep_regressed(round_grms_start, float(grms)):
            if not quiet:
                print(
                    f"round{round_idx + 1}: rollback entire round "
                    f"(GRMS {round_grms_start:.1f} -> {float(grms):.1f} kcal/mol/Å)",
                    flush=True,
                )
            grms = _rollback_charmm_geometry(round_pos_start, mlpot_ctx, quiet=quiet)

    result.final_grms = float(grms)
    result.reason = "grms_ok" if grms <= max_grms else "grms_still_high"
    journal.finish(float(grms), reason=result.reason)
    if progress is not None:
        progress.finish(result.to_dict())
    return float(grms), box_side, result


def maybe_run_density_prep_ladder_for_mlpot(
    mlpot_ctx: Any,
    *,
    max_grms: float,
    context: str = "Density prep ladder",
    quiet: bool = False,
) -> tuple[float, bool]:
    """Run the liquid-prep ladder when enabled on ``mlpot_ctx.workflow_args``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import refresh_mlpot_energy_and_grms

    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is None or not density_prep_ladder_enabled(args):
        return refresh_mlpot_energy_and_grms(mlpot_ctx, context=context), False

    atoms_per_list = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per_list is None:
        atoms_per_list = getattr(args, "_cluster_atoms_per_list", None)
    if atoms_per_list is None:
        return refresh_mlpot_energy_and_grms(mlpot_ctx, context=context), False

    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if pyCModel is None:
        return refresh_mlpot_energy_and_grms(mlpot_ctx, context=context), False

    n_mol = len(atoms_per_list)
    n_atoms = int(sum(atoms_per_list))
    box_side = getattr(mlpot_ctx, "cubic_box_side_A", None)
    if box_side is None:
        box_side = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)
    composition = getattr(args, "_cluster_composition_summary", None)
    current_grms = refresh_mlpot_energy_and_grms(
        mlpot_ctx,
        context=f"{context} (initial)" if not quiet else "",
    )
    if current_grms <= float(max_grms):
        return float(current_grms), False

    ladder_grms, _new_side, _summary = run_density_prep_ladder(
        args,
        mlpot_ctx=mlpot_ctx,
        pyCModel=pyCModel,
        max_grms=float(max_grms),
        current_grms=float(current_grms),
        n_mol=n_mol,
        n_atoms=n_atoms,
        box_side=float(box_side) if box_side is not None else None,
        charmm_pbc=bool(getattr(mlpot_ctx, "use_pbc", False)),
        atoms_per_list=list(atoms_per_list),
        composition=composition,
        mini_nstep=int(getattr(args, "mini_nstep", 500) or 500),
        mini_nprint=max(1, int(getattr(args, "nprint", 100) or 100)),
        fix_resids=[],
        show_energy=False,
        z=getattr(mlpot_ctx, "ml_Z", None),
    )
    return float(ladder_grms), True


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
    atomic_numbers: np.ndarray | None = None,
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
    lattice_steps = resolve_density_prep_lattice_abnr_steps(args)
    staged_fraction = float(getattr(args, "liquid_prep_staged_density_fraction", 0.70) or 0.70)

    result = PreMlpotGeometryGateResult(
        enabled=True,
        ran=True,
        reason="running",
    )
    pos = np.asarray(positions, dtype=np.float64)
    side = box_side
    from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
        RecoveryProgressStore,
    )

    progress = RecoveryProgressStore.for_prep_ladder(
        args,
        title="Pre-MLpot geometry gate",
        quiet=quiet,
    )

    def _record_gate_step(label: str, *, note: str = "") -> None:
        if progress is None:
            return
        from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

        progress.record_step(
            label,
            box_side_A=side,
            positions=get_charmm_positions_array(),
            note=note,
        )

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
    _record_gate_step("initial", note=f"worst inter-monomer {result.worst_intermonomer_A:.3f} Å")

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
        _record_gate_step(step_label)
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
                _record_gate_step(step_label)
            except Exception as exc:
                if not quiet:
                    print(f"Pre-MLpot gate: skip {step_label} ({exc})", flush=True)

    if charmm_pbc and lattice_steps > 0:
        for nocoords, tag in ((False, "lattice_full"), (True, "lattice_box")):
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
                    allow_prepare_pbc=True,
                )
                if new_side is not None:
                    side = float(new_side)
                result.steps_applied.append(step_label)
                _record_gate_step(step_label)
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
        from mmml.utils.intermonomer_geometry import (
            resolve_dynamics_overlap_reference_A,
            summarize_worst_intermonomer_contact,
        )

        z_arr = atomic_numbers
        if z_arr is None:
            z_arr = getattr(args, "_cluster_atomic_numbers", None)
        summary = summarize_worst_intermonomer_contact(
            pos,
            atoms_per_list,
            box_side=side,
            use_pbc=charmm_pbc,
            threshold_A=min_overlap,
            atomic_numbers=z_arr,
            dynamics_reference_A=resolve_dynamics_overlap_reference_A(args),
        )
        print(
            f"Pre-MLpot geometry gate: {len(result.steps_applied)} step(s), "
            f"{summary.format_log_line()}",
            flush=True,
        )
        if progress is not None:
            progress.finish(
                {
                    **result.to_dict(),
                    "worst_intermonomer_A": float(result.worst_intermonomer_A or 0.0),
                    "contact_summary": summary.format_log_line(),
                }
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
        measure_hybrid_charmm_grms,
        refresh_mlpot_energy_and_grms,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array
    from mmml.utils.prep_ladder_report import PrepLadderJournal, PrepMetrics

    quiet = not verbose
    min_overlap = resolve_pre_mlpot_overlap_min_distance(args)
    spacing = getattr(args, "spacing", None)
    seed = getattr(args, "seed", None)
    journal = PrepLadderJournal(title=context_prefix, quiet=quiet)
    from mmml.interfaces.pycharmmInterface.mlpot.recovery_progress import (
        RecoveryProgressStore,
    )

    progress = RecoveryProgressStore.for_cleanup(
        args,
        title=str(context_prefix),
        quiet=quiet,
    )

    def _metrics(step_grms: float) -> PrepMetrics:
        diag = measure_hybrid_charmm_grms(mlpot_ctx)
        return PrepMetrics.from_mlpot(
            mlpot_ctx,
            hybrid_grms=float(step_grms),
            charmm_grms=float(diag.charmm),
            diag_kind=str(diag.kind),
        )

    grms = refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
    pos_initial = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
    initial_grms = float(grms)
    journal.begin(initial_grms=grms, max_grms=float(grms_limit or grms), max_rounds=1)
    journal.record_step("initial", _metrics(grms))
    if progress is not None:
        progress.record_step(
            "initial",
            grms_kcalmol_A=float(grms),
            box_side_A=box_side,
            positions=pos_initial,
        )

    def _record_cleanup_step(label: str, step_grms: float) -> None:
        if progress is None:
            return
        progress.record_step(
            label,
            grms_kcalmol_A=float(step_grms),
            box_side_A=box_side,
            positions=get_charmm_positions_array(),
        )

    step_label = f"{context_prefix}:monomer_repack"
    try:
        pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
        grms_before = float(grms)
        pos = pos_before
        new_pos = _step_monomer_repack(
            pos,
            atoms_per_list=list(atoms_per_list),
            box_side=box_side,
            min_distance=min_overlap,
            spacing=float(spacing) if spacing is not None else None,
            seed=int(seed) if seed is not None else None,
            mlpot_ctx=mlpot_ctx,
            verbose=verbose,
        )
        box_side = _sync_pbc_after_box_change(
            positions=new_pos,
            box_side=box_side,
            charmm_pbc=charmm_pbc,
            mlpot_ctx=mlpot_ctx,
            args=args,
            quiet=quiet,
            report_resync=False,
        )
        grms_after = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
        if _geometry_prep_regressed(grms_before, grms_after):
            if verbose:
                print(
                    f"{context_prefix}: rollback monomer_repack "
                    f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                    flush=True,
                )
            grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
            journal.skip_step(
                step_label,
                f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
            )
        else:
            grms = grms_after
            journal.record_step(step_label, _metrics(grms))
            _record_cleanup_step(step_label, grms)
    except Exception as exc:
        journal.skip_step(step_label, str(exc))

    if charmm_pbc and composition is not None:
        step_label = f"{context_prefix}:mc_density"
        try:
            pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
            grms_before = float(grms)
            pos = pos_before
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
                report_resync=False,
            )
            grms_after = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
            if _geometry_prep_regressed(grms_before, grms_after):
                if verbose:
                    print(
                        f"{context_prefix}: rollback mc_density "
                        f"(GRMS {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å)",
                        flush=True,
                    )
                grms = _rollback_charmm_geometry(pos_before, mlpot_ctx, quiet=quiet)
                journal.skip_step(
                    step_label,
                    f"GRMS regressed {grms_before:.1f} -> {grms_after:.1f} kcal/mol/Å",
                )
            else:
                grms = grms_after
                journal.record_step(step_label, _metrics(grms))
                _record_cleanup_step(step_label, grms)
        except Exception as exc:
            journal.skip_step(step_label, str(exc))

    if calculator_minimize:
        from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
            HybridCalculatorFireConfig,
            HybridCalculatorMinimizeConfig,
            coerce_hybrid_minimize_result,
            minimize_hybrid_calculator_before_sd,
            minimize_hybrid_calculator_fire_before_sd,
            resolve_calculator_mini_safe_grms,
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
        safe_grms = resolve_calculator_mini_safe_grms(
            args=args,
            context="geometry_packing",
        )
        fire_config = HybridCalculatorFireConfig(
            max_steps=int(calculator_fire_steps),
            fmax_ev_a=fire_fmax,
            fire_maxstep=float(calculator_fire_maxstep),
            verbose=verbose,
            max_start_grms_kcalmol_A=start_cap,
            safe_grms_kcalmol_A=safe_grms,
        )
        bfgs_config = HybridCalculatorMinimizeConfig(
            max_steps=int(calculator_minimize_steps),
            fmax_ev_a=float(calculator_minimize_fmax_ev_a),
            bfgs_maxstep=float(calculator_bfgs_maxstep),
            verbose=verbose,
            quiet_bfgs=quiet_bfgs,
            max_start_grms_kcalmol_A=start_cap,
            max_initial_fmax_ev_a=1000.0,
            safe_grms_kcalmol_A=safe_grms,
        )

        def _run_bfgs() -> float:
            nonlocal grms
            skip_below = float(
                getattr(args, "geometry_packing_skip_bfgs_grms", 5.0) or 0.0
            )
            if skip_below > 0.0 and np.isfinite(grms) and float(grms) <= skip_below:
                if verbose:
                    print(
                        f"{context_prefix} (BFGS): skip "
                        f"(GRMS {float(grms):.2f} <= {skip_below:.1f} kcal/mol/Å)",
                        flush=True,
                    )
                return float(grms)
            mini = coerce_hybrid_minimize_result(
                minimize_hybrid_calculator_before_sd(
                    mlpot_ctx,
                    bfgs_config,
                    context_prefix=f"{context_prefix} (BFGS)",
                )
            )
            grms = float(mini.grms)
            if not mini.ran:
                return refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
            refreshed = refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
            journal.record_step(f"{context_prefix} (post-BFGS)", _metrics(refreshed))
            _record_cleanup_step(f"{context_prefix} (post-BFGS)", refreshed)
            return refreshed

        def _run_fire() -> float:
            nonlocal grms
            fire = coerce_hybrid_minimize_result(
                minimize_hybrid_calculator_fire_before_sd(
                    mlpot_ctx,
                    config=fire_config,
                    context_prefix=f"{context_prefix} (FIRE)",
                )
            )
            grms = float(fire.grms)
            if not fire.ran:
                return refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
            refreshed = refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
            journal.record_step(f"{context_prefix} (post-FIRE)", _metrics(refreshed))
            _record_cleanup_step(f"{context_prefix} (post-FIRE)", refreshed)
            return refreshed

        if verbose and bfgs_first:
            from mmml.utils.prep_ladder_report import emit_prep_phase

            emit_prep_phase(
                context_prefix,
                "guarded BFGS before FIRE",
                metrics=_metrics(grms),
                note=f"GRMS {grms:.1f} > {fire_bfgs_crossover:.1f}",
                quiet=quiet,
            )

        runners = (_run_bfgs, _run_fire) if bfgs_first else (_run_fire, _run_bfgs)
        for runner in runners:
            try:
                grms = runner()
            except Exception as exc:
                journal.skip_step(f"{context_prefix} calculator mini", str(exc))

    if (
        grms_limit is not None
        and float(grms) > float(grms_limit)
        and getattr(mlpot_ctx, "pyCModel", None) is not None
    ):
        step_label = f"{context_prefix}:mlpot_sd"
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                _run_mlpot_recovery_mini,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                BondedMmMiniConfig,
            )

            if verbose:
                from mmml.utils.prep_ladder_report import emit_prep_phase

                emit_prep_phase(
                    context_prefix,
                    "MLpot SD recovery",
                    metrics=_metrics(grms),
                    note=(
                        f"GRMS {float(grms):.1f} still > {float(grms_limit):.1f} "
                        "after repack/calculator; running short MLpot SD"
                    ),
                    quiet=quiet,
                )
            _run_mlpot_recovery_mini(
                mlpot_ctx,
                BondedMmMiniConfig(
                    nstep_sd=int(getattr(args, "bonded_mm_mini_steps", 200) or 200),
                    nprint=max(1, int(getattr(args, "nprint", 10) or 10)),
                    verbose=verbose,
                ),
                pyCModel=mlpot_ctx.pyCModel,
                context=str(step_label),
                calculator_pre_minimize=False,
            )
            grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
            journal.record_step(step_label, _metrics(grms))
            _record_cleanup_step(step_label, grms)
        except Exception as exc:
            journal.skip_step(step_label, str(exc))

    if _geometry_prep_regressed(initial_grms, float(grms), ratio=1.05, min_delta=10.0):
        if verbose:
            print(
                f"{context_prefix}: rollback to pre-packing geometry "
                f"(final GRMS {float(grms):.1f} > initial {initial_grms:.1f} kcal/mol/Å)",
                flush=True,
            )
        grms = _rollback_charmm_geometry(pos_initial, mlpot_ctx, quiet=quiet)

    journal.finish(float(grms), reason="packing_done")
    if progress is not None:
        progress.finish(
            {
                "initial_grms": float(initial_grms),
                "final_grms": float(grms),
                "context_prefix": str(context_prefix),
            }
        )
    return float(grms)
