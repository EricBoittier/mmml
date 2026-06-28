"""Unit tests for resilient density/box preparation ladder."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


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


def test_condensed_phase_defaults_from_certified_box():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_condensed_phase_md_defaults,
        density_prep_ladder_enabled,
        liquid_prep_enabled,
    )

    args = _args(from_psf="boxes/dcm52/model.psf", skip_cluster_build=True)
    apply_condensed_phase_md_defaults(args)
    assert liquid_prep_enabled(args)
    assert density_prep_ladder_enabled(args)
    assert args.mini_box_equil_ps == 0.0
    assert args.mini_lattice_abnr_steps == 0
    assert args.calculator_pre_minimize is True
    assert int(args.fire_min_steps) >= 200
    assert int(args.pre_min_steps) >= 200


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


def test_sync_pbc_after_box_change_skips_prepare_charmm_pbc_with_mlpot(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        _sync_pbc_after_box_change,
    )

    calls: list[str] = []

    class _Ctx:
        cubic_box_side_A = 28.0
        charmm_cubic_box_side_A = 28.0
        pyCModel = object()
        use_pbc = True

    def _fake_prepare(_side: float) -> None:
        calls.append("prepare_charmm_pbc")

    def _fake_light_resync(*_a, **_kw) -> float:
        calls.append("light_resync")
        return 1.0

    def _fake_sync_workflow(*_a, **_kw) -> float:
        calls.append("sync_workflow")
        return 30.0

    def _fake_sync_mic(*_a, **_kw) -> float:
        calls.append("sync_mic")
        return 30.0

    def _fake_sync_pos(_pos) -> None:
        calls.append("sync_pos")

    def _fake_crystal_active() -> bool:
        return False

    def _fake_sync_crystal(_side: float, *, quiet: bool = False) -> bool:
        calls.append("sync_crystal")
        return True

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.prepare_charmm_pbc",
        _fake_prepare,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        _fake_crystal_active,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.sync_charmm_crystal_after_mm_pretreat",
        _fake_sync_crystal,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.light_resync_mlpot_state",
        _fake_light_resync,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.sync_workflow_pbc_box_side_after_mm_pretreat",
        _fake_sync_workflow,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.sync_mlpot_pbc_cell_from_charmm",
        _fake_sync_mic,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        _fake_sync_pos,
    )

    side = _sync_pbc_after_box_change(
        positions=np.zeros((4, 3)),
        box_side=30.0,
        charmm_pbc=True,
        mlpot_ctx=_Ctx(),
        quiet=True,
    )
    assert side == 30.0
    assert "prepare_charmm_pbc" not in calls
    assert "sync_crystal" in calls
    assert "light_resync" in calls
    assert "sync_mic" in calls


def test_geometry_prep_regressed_detects_large_spike():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        _geometry_prep_regressed,
    )

    assert _geometry_prep_regressed(574.0, 5917.0)
    assert not _geometry_prep_regressed(574.0, 600.0)
    assert _geometry_prep_regressed(10.0, 70.0)


def test_run_geometry_packing_recovery_rolls_back_repack_grms_regression(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        run_geometry_packing_recovery,
    )

    args = _args(quiet=True)
    pos = np.arange(24, dtype=float).reshape(8, 3)
    sync_log: list[np.ndarray] = []
    refresh_values = iter([574.0, 5917.0, 574.0, 574.0])

    class _Diag:
        hybrid = 1.0
        charmm = 1.0
        kind = "geometry_stress"

    class _Ctx:
        use_pbc = True
        cubic_box_side_A = 29.0
        charmm_cubic_box_side_A = 29.0
        pyCModel = object()

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: pos.copy(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder._step_monomer_repack",
        lambda *_a, **_kw: pos + 5.0,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder._sync_pbc_after_box_change",
        lambda **_kw: 29.0,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        lambda arr: sync_log.append(np.asarray(arr, dtype=float).copy()),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        lambda *_a, **_kw: float(next(refresh_values)),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        lambda *_a, **_kw: _Diag(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize.minimize_hybrid_calculator_before_sd",
        lambda *_a, **_kw: (574.0, True),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize.minimize_hybrid_calculator_fire_before_sd",
        lambda *_a, **_kw: (574.0, True),
    )

    final_grms = run_geometry_packing_recovery(
        _Ctx(),
        args=args,
        atoms_per_list=[4, 4],
        composition={"DCM": 2},
        box_side=29.0,
        charmm_pbc=True,
        context_prefix="Pre-SD packing",
        calculator_minimize=False,
        verbose=False,
        grms_limit=574.0,
    )
    assert final_grms == pytest.approx(574.0)
    assert len(sync_log) >= 1
    assert np.allclose(sync_log[-1], pos)


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
