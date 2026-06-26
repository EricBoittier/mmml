"""Unit tests for resilient density/box preparation ladder."""

from __future__ import annotations

import argparse

import numpy as np


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        liquid_prep=False,
        density_prep_mode="off",
        density_prep_ladder=None,
        density_prep_ladder_max_rounds=3,
        density_prep_lattice_abnr_steps=0,
        box_size=None,
        target_density_g_cm3=None,
        bulk_density_fraction=None,
        mc_density_equalize=True,
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        mini_nstep=20,
        bonded_mm_mini_steps=200,
        mini_lattice_abnr_steps=0,
        mini_box_equil_ps=0.0,
        mini_lattice_abnr_allow_fixed_box=False,
        mini_box_equil_allow_fixed_box=False,
        calculator_pre_minimize=True,
        quiet=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resilient_defaults_bump_mini_and_enable_ladder():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
        density_prep_ladder_enabled,
    )

    args = _args(density_prep_mode="resilient")
    apply_density_prep_resilient_defaults(args)
    assert density_prep_ladder_enabled(args)
    assert args.bulk_density_fraction == 0.75
    assert args.charmm_sd_steps == 1000
    assert args.charmm_abnr_steps == 1000
    assert args.mini_nstep == 500
    assert args.mini_lattice_abnr_steps == 200
    assert args.mini_box_equil_ps == 2.0


def test_resilient_defaults_respect_explicit_box_and_ladder_off():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
        density_prep_ladder_enabled,
    )

    args = _args(
        density_prep_mode="resilient",
        density_prep_ladder=False,
        box_size=32.0,
        bulk_density_fraction=0.5,
        charmm_sd_steps=2000,
    )
    apply_density_prep_resilient_defaults(args)
    assert not density_prep_ladder_enabled(args)
    assert args.bulk_density_fraction == 0.5
    assert args.charmm_sd_steps == 2000
    assert args.mini_lattice_abnr_allow_fixed_box is True
    assert args.mini_box_equil_allow_fixed_box is True


def test_ladder_skipped_when_grms_ok():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        run_density_prep_ladder,
    )

    args = _args(density_prep_mode="resilient", density_prep_ladder=True)
    grms, side, summary = run_density_prep_ladder(
        args,
        mlpot_ctx=object(),
        pyCModel=object(),
        max_grms=50.0,
        current_grms=10.0,
        n_mol=2,
        n_atoms=20,
        box_side=28.0,
        charmm_pbc=True,
        atoms_per_list=[10, 10],
        composition={"DCM": 2},
        mini_nstep=100,
        mini_nprint=10,
        fix_resids=[],
        show_energy=False,
        z=np.ones(20, dtype=int),
    )
    assert grms == 10.0
    assert side == 28.0
    assert summary.ran is False
    assert summary.reason == "grms_ok"


def test_liquid_prep_shorthand_enables_defaults():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
        density_prep_ladder_enabled,
        liquid_prep_enabled,
    )

    args = _args(liquid_prep=True)
    assert liquid_prep_enabled(args)
    apply_density_prep_resilient_defaults(args)
    assert density_prep_ladder_enabled(args)
    assert args.mini_lattice_abnr_steps == 200


def test_build_pycharmm_command_forwards_liquid_prep():
    from mmml.cli.run.md_system import build_pycharmm_command
    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    cmd = build_pycharmm_command(_pycharmm_args(liquid_prep=True))
    assert "--liquid-prep" in cmd
    assert "--density-prep-mode" not in cmd


def test_build_pycharmm_command_forwards_density_prep_flags():
    from mmml.cli.run.md_system import build_pycharmm_command
    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    cmd = build_pycharmm_command(
        _pycharmm_args(
            density_prep_mode="resilient",
            density_prep_ladder=True,
            density_prep_ladder_max_rounds=5,
            density_prep_lattice_abnr_steps=150,
        )
    )
    assert "--density-prep-mode" in cmd
    assert "resilient" in cmd
    assert "--density-prep-ladder" in cmd
    assert "--density-prep-ladder-max-rounds" in cmd
    assert "5" in cmd
    assert "--density-prep-lattice-abnr-steps" in cmd
    assert "150" in cmd
