"""Unit tests for unified --cleanup flag."""

from __future__ import annotations

import argparse


def _base_args(**overrides) -> argparse.Namespace:
    base = dict(
        cleanup=False,
        liquid_prep=False,
        density_prep_mode="off",
        density_prep_ladder=None,
        density_prep_ladder_max_rounds=1,
        bonded_mm_mini=True,
        calculator_pre_minimize=True,
        dynamics_overlap_action="warn",
        dynamics_overlap_charmm_sd_steps=100,
        dynamics_overlap_charmm_abnr_steps=100,
        bonded_mm_mini_steps=50,
        pre_min_steps=50,
        charmm_sd_steps=50,
        charmm_abnr_steps=50,
        mini_nstep=50,
        mini_lattice_abnr_steps=0,
        mini_box_equil_ps=0.0,
        save_run_state=False,
        mc_density_equalize=True,
        no_dynamics_overlap_separate=False,
        quiet=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_apply_cleanup_defaults_enables_recovery_stack():
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import (
        apply_cleanup_defaults,
        cleanup_enabled,
        cleanup_ladder_enabled,
        cleanup_overlap_fallback_enabled,
        cleanup_prep_enabled,
    )

    args = _base_args(cleanup=True)
    apply_cleanup_defaults(args)
    assert cleanup_enabled(args)
    assert args.liquid_prep is True
    assert args.density_prep_ladder is True
    assert args.dynamics_overlap_action == "rescue"
    assert args.calculator_pre_minimize is True
    assert args.bonded_mm_mini is True
    assert args.save_run_state is True
    assert args.charmm_sd_steps >= 1000
    assert args.dynamics_overlap_charmm_sd_steps >= 400
    assert args.pre_min_steps >= 300
    assert cleanup_prep_enabled(args)
    assert cleanup_ladder_enabled(args)
    assert cleanup_overlap_fallback_enabled(args)


def test_apply_cleanup_defaults_noop_when_disabled():
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import apply_cleanup_defaults

    args = _base_args(cleanup=False, dynamics_overlap_action="warn")
    apply_cleanup_defaults(args)
    assert args.liquid_prep is False
    assert args.dynamics_overlap_action == "warn"
    assert args.save_run_state is False


def test_build_pycharmm_command_forwards_cleanup():
    from mmml.cli.run.md_system import build_pycharmm_command
    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    cmd = build_pycharmm_command(_pycharmm_args(cleanup=True))
    assert "--cleanup" in cmd
    assert "--liquid-prep" not in cmd


def test_overlap_config_uses_cleanup_for_ladder_fallback():
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import (
        cleanup_overlap_fallback_enabled,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        resolve_dynamics_overlap_config,
    )

    args = _base_args(cleanup=True)
    assert cleanup_overlap_fallback_enabled(args)
    cfg = resolve_dynamics_overlap_config(
        args,
        n_monomers=10,
        use_pbc=True,
        fallback_box_side_A=29.0,
    )
    assert cfg.density_prep_ladder_fallback is True
    assert cfg.cleanup_mode is True
