"""Unified ``--cleanup`` flag for geometry recovery across prep and dynamics."""

from __future__ import annotations

import argparse
from typing import Any

from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
    _bump_int_attr,
    apply_density_prep_resilient_defaults,
    density_prep_ladder_enabled,
    liquid_prep_enabled,
)


def add_cleanup_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Geometry cleanup (one-shot recovery)")
    group.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the full geometry cleanup ladder: resilient liquid prep, "
            "pre-MLpot repack gate, density prep ladder, hybrid calculator "
            "pre-minimize, bonded-MM recovery, and dynamics overlap rescue "
            "(selective monomer repack when forces indicate 1–2 hot spots). "
            "Use once when a run breaks (ECHECK, overlap, high GRMS) to reach a "
            "stable restart, then re-run without --cleanup for production "
            "trajectories where time-series correlations matter. "
            "Superset of --liquid-prep; individual recovery flags remain overridable."
        ),
    )


def cleanup_enabled(args: argparse.Namespace | Any) -> bool:
    """True when the unified cleanup ladder should run."""
    return bool(getattr(args, "cleanup", False))


def apply_cleanup_defaults(args: argparse.Namespace) -> None:
    """Apply recovery defaults when ``--cleanup`` is set (idempotent)."""
    if not cleanup_enabled(args):
        return

    args.liquid_prep = True
    apply_density_prep_resilient_defaults(args)

    args.density_prep_ladder = True
    args.calculator_pre_minimize = bool(
        getattr(args, "calculator_pre_minimize", True) is not False
    )
    args.bonded_mm_mini = bool(getattr(args, "bonded_mm_mini", True) is not False)

    action = str(getattr(args, "dynamics_overlap_action", "rescue") or "rescue").lower()
    if action in ("off", "warn", "error"):
        args.dynamics_overlap_action = "rescue"

    if getattr(args, "no_dynamics_overlap_separate", False):
        args.no_dynamics_overlap_separate = False

    _bump_int_attr(args, "dynamics_overlap_charmm_sd_steps", 400)
    _bump_int_attr(args, "dynamics_overlap_charmm_abnr_steps", 400)
    _bump_int_attr(args, "bonded_mm_mini_steps", 200)
    _bump_int_attr(args, "pre_min_steps", 300)
    if hasattr(args, "fire_min_steps"):
        _bump_int_attr(args, "fire_min_steps", 300)
    _bump_int_attr(args, "density_prep_ladder_max_rounds", 3)

    if not bool(getattr(args, "save_run_state", False)):
        args.save_run_state = True


def cleanup_prep_enabled(args: argparse.Namespace | Any) -> bool:
    """True when preventive prep / ladder paths tied to cleanup should run."""
    return cleanup_enabled(args) or liquid_prep_enabled(args)


def cleanup_ladder_enabled(args: argparse.Namespace | Any) -> bool:
    """True when post-mini density / geometry ladder should run."""
    if cleanup_enabled(args):
        return True
    return density_prep_ladder_enabled(args)


def cleanup_overlap_fallback_enabled(args: argparse.Namespace | Any) -> bool:
    """True when extent fly-off may fall back to the density prep ladder."""
    if cleanup_enabled(args):
        return True
    if liquid_prep_enabled(args):
        return True
    mode = str(getattr(args, "density_prep_mode", "off") or "off").lower()
    return mode == "resilient"
