"""Tests for bonded-MM recovery helpers."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest
import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import MmStrainBaseline
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_internal_energy_kcalmol


def test_charmm_internal_energy_prefers_inte():
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._charmm_eterm_value",
        side_effect=lambda name: {"INTE": 12.5, "BOND": 1.0}.get(name.upper()),
    ):
        assert charmm_internal_energy_kcalmol() == pytest.approx(12.5)


def test_charmm_internal_energy_sums_bonded_terms():
    def fake_eterm(name: str):
        return {"BOND": 1.0, "ANGL": 2.0, "DIHE": 0.5}.get(name.upper())

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._charmm_eterm_value",
        side_effect=fake_eterm,
    ):
        assert charmm_internal_energy_kcalmol() == pytest.approx(3.5)


def test_charmm_internal_energy_prefers_bonded_when_inte_zero():
    def fake_eterm(name: str):
        return {"INTE": 0.0, "BOND": 10.0, "ANGL": 2.0}.get(name.upper())

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._charmm_eterm_value",
        side_effect=fake_eterm,
    ):
        assert charmm_internal_energy_kcalmol() == pytest.approx(12.0)


def test_bonded_recovery_sd_kwargs_pbc_vs_vacuum():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        BondedMmMiniConfig,
        _bonded_recovery_sd_kwargs,
    )

    cfg = BondedMmMiniConfig(nstep_sd=10)
    pbc_ctx = MagicMock(use_pbc=True)
    vac_ctx = MagicMock(use_pbc=False)
    pbc_kw = _bonded_recovery_sd_kwargs(pbc_ctx, cfg)
    vac_kw = _bonded_recovery_sd_kwargs(vac_ctx, cfg)
    assert pbc_kw["nstep"] == 10
    assert pbc_kw["inbfrq"] == 50
    assert pbc_kw["ihbfrq"] == 50
    assert "imgfrq" not in pbc_kw
    assert vac_kw["inbfrq"] == 50
    assert vac_kw["ihbfrq"] == 0
    assert "imgfrq" not in vac_kw


def test_charmm_bonded_term_reads_angl():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import charmm_bonded_term_kcalmol

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._charmm_eterm_value",
        return_value=42.0,
    ) as eterm:
        assert charmm_bonded_term_kcalmol("ANGL") == pytest.approx(42.0)
    eterm.assert_called_once_with("ANGL")


def test_apply_bonded_mm_only_block_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/mlpot/block_terms.py"
    )
    spec = importlib.util.spec_from_file_location("block_terms", path)
    block_terms = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(block_terms)

    scripts: list[str] = []
    with patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
        side_effect=scripts.append,
    ):
        block_terms.apply_bonded_mm_only_block()
    script = scripts[0]
    assert "BOND 1.0 ANGL 1.0 DIHEdral 1.0" in script
    assert "ELEC 0.0 VDW 0.0" in script


def test_apply_bonded_vdw_recovery_block_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/mlpot/block_terms.py"
    )
    spec = importlib.util.spec_from_file_location("block_terms", path)
    block_terms = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(block_terms)

    scripts: list[str] = []
    with patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
        side_effect=scripts.append,
    ):
        block_terms.apply_bonded_vdw_recovery_block()
    script = scripts[0]
    assert "BOND 1.0 ANGL 1.0 DIHEdral 1.0" in script
    assert "ELEC 0.0 VDW 1.0" in script


def test_flat_bottom_mmfp_uses_outside_harmonic_wall():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    pycharmm = MagicMock()
    with patch.object(restraints, "_import_pycharmm", return_value=pycharmm):
        restraints.setup_flat_bottom_sphere_mmfp(
            restraints.FlatBottomSphereConfig(
                radius=10.0,
                force=0.01,
                selection="TYPE C",
            )
        )

    script = pycharmm.lingo.charmm_script.call_args[0][0]
    assert "GEO sphere harm" in script
    assert "droff 10.000000 force 0.010000" in script
    assert " outside " not in script
    assert " inside " not in script
    assert "sele TYPE C end" in script
    assert "quartic" not in script.lower()


def test_apply_flat_bottom_workflow_accepts_selection():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin") as center, patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ) as setup, patch.object(
        restraints,
        "_selected_max_radius",
        return_value=4.0,
    ), patch.object(
        restraints,
        "_current_charmm_energy_kcalmol",
        return_value=None,
    ):
        cfg = restraints.apply_flat_bottom_workflow(
            radius=10.0,
            force=0.01,
            selection="TYPE CG321",
        )

    center.assert_called_once()
    setup.assert_called_once()
    assert cfg is not None
    assert cfg.selection == "TYPE CG321"


def test_apply_flat_bottom_workflow_inflates_droff_to_current_extent():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin"), patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ) as setup, patch.object(
        restraints,
        "_selected_max_radius",
        return_value=12.5,
    ), patch.object(
        restraints,
        "_current_charmm_energy_kcalmol",
        return_value=None,
    ):
        cfg = restraints.apply_flat_bottom_workflow(
            radius=10.0,
            force=0.01,
            selection="TYPE C",
        )

    assert cfg is not None
    assert cfg.radius > 12.5
    assert cfg.radius == pytest.approx(12.501)
    setup.assert_called_once()
    assert setup.call_args.args[0].radius == pytest.approx(12.501)


def test_selected_max_radius_uses_charmm_selection_bounds():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    pycharmm = MagicMock()
    values = {
        "XMIN": -3.0,
        "XMAX": 4.0,
        "YMIN": -5.0,
        "YMAX": 2.0,
        "ZMIN": -1.0,
        "ZMAX": 6.0,
    }
    pycharmm.lingo.get_energy_value.side_effect = lambda key: values[key.upper()]

    with patch.object(restraints, "_import_pycharmm", return_value=pycharmm):
        radius = restraints._selected_max_radius(
            "TYPE C",
            xref=0.0,
            yref=0.0,
            zref=0.0,
        )

    pycharmm.lingo.charmm_script.assert_called_once_with("coor stat sele TYPE C end")
    assert radius == pytest.approx((4.0**2 + 5.0**2 + 6.0**2) ** 0.5)


def test_apply_flat_bottom_workflow_verifies_energy_unchanged():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin"), patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ), patch.object(
        restraints,
        "_selected_max_radius",
        return_value=8.0,
    ), patch.object(
        restraints,
        "_current_charmm_energy_kcalmol",
        side_effect=[100.0, 100.0],
    ) as energy, patch(
        "builtins.print",
    ) as mock_print:
        restraints.apply_flat_bottom_workflow(radius=10.0, force=0.01, selection="TYPE C")

    assert energy.call_count == 2
    assert any(
        "zero-energy check OK" in str(call.args[0])
        for call in mock_print.call_args_list
    )


def test_apply_flat_bottom_workflow_retries_until_energy_unchanged():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin"), patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ) as setup, patch.object(
        restraints,
        "_selected_max_radius",
        return_value=8.0,
    ), patch.object(
        restraints,
        "_current_charmm_energy_kcalmol",
        side_effect=[100.0, 100.01, 100.0],
    ), patch(
        "builtins.print",
    ) as mock_print:
        cfg = restraints.apply_flat_bottom_workflow(
            radius=10.0,
            force=0.01,
            selection="TYPE C",
        )

    assert cfg is not None
    assert cfg.radius > 10.0
    assert setup.call_count == 2
    assert any(
        "increasing droff" in str(call.args[0])
        for call in mock_print.call_args_list
    )
    assert any(
        "zero-energy check OK" in str(call.args[0])
        for call in mock_print.call_args_list
    )


def test_apply_flat_bottom_workflow_warns_when_energy_never_converges():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin"), patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ) as setup, patch.object(
        restraints,
        "_selected_max_radius",
        return_value=8.0,
    ), patch.object(
        restraints,
        "_current_charmm_energy_kcalmol",
        side_effect=[100.0, 100.01, 100.02],
    ), patch.object(
        restraints,
        "_DROFF_TUNE_MAX_ATTEMPTS",
        2,
    ), patch(
        "builtins.print",
    ) as mock_print:
        restraints.apply_flat_bottom_workflow(radius=10.0, force=0.01, selection="TYPE C")

    assert setup.call_count == 2
    assert any(
        "WARN: MMFP flat-bottom changed energy" in str(call.args[0])
        for call in mock_print.call_args_list
    )


def test_clear_mmfp_uses_block_command():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    pycharmm = MagicMock()
    with patch.object(restraints, "_import_pycharmm", return_value=pycharmm), patch.object(
        restraints,
        "_MMFP_GEO_ACTIVE",
        True,
    ):
        restraints.clear_mmfp_restraints()

    script = pycharmm.lingo.charmm_script.call_args[0][0]
    assert "MMFP\nGEO RESET\nEND" in "\n".join(
        line.strip() for line in script.splitlines() if line.strip()
    )
    assert "CLEAR" not in script
    assert "MMFP CLEAR" not in script


def test_clear_mmfp_noops_before_setup():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    pycharmm = MagicMock()
    with patch.object(restraints, "_import_pycharmm", return_value=pycharmm), patch.object(
        restraints,
        "_MMFP_GEO_ACTIVE",
        False,
    ):
        restraints.clear_mmfp_restraints()

    pycharmm.lingo.charmm_script.assert_not_called()


def test_minimize_bonded_recovery_uses_bonded_only_block():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        BondedMmMiniConfig,
        minimize_bonded_mm_recovery,
    )

    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = False
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_detached",
        side_effect=lambda _ctx, fn: fn(),
    ) as detached, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_bonded_mm_only_block",
    ) as bonded_block, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_bonded_mm_rescue_environment",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
    ):
        imp.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        minimize_bonded_mm_recovery(ctx, BondedMmMiniConfig(nstep_sd=0))
    detached.assert_called_once()
    bonded_block.assert_called_once()


def test_minimize_bonded_recovery_runs_sd_and_reports_angl():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        BondedMmMiniConfig,
        minimize_bonded_mm_recovery,
    )

    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = False
    minimize = MagicMock()
    angl_values = iter([500.0, 50.0])

    def fake_eterm(name: str):
        if name.upper() == "ANGL":
            return next(angl_values)
        return None

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_detached",
        side_effect=lambda _ctx, fn: fn(),
    ) as detached, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._log_bonded_term_diagnostics",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_bonded_mm_only_block",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_bonded_mm_rescue_environment",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.charmm_internal_energy_kcalmol",
        return_value=550.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._charmm_eterm_value",
        side_effect=fake_eterm,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=MagicMock(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
    ):
        imp.return_value = (MagicMock(), MagicMock(), MagicMock(), minimize)
        grms = minimize_bonded_mm_recovery(ctx, BondedMmMiniConfig(nstep_sd=30))
    detached.assert_called_once_with(ctx, ANY)
    minimize.run_sd.assert_called_once()
    assert minimize.run_sd.call_args.kwargs["nstep"] == 30
    assert grms == pytest.approx(1.0)


def test_minimize_bonded_recovery_unset_and_reregister():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        BondedMmMiniConfig,
        minimize_bonded_mm_recovery,
    )

    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = False
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_bonded_mm_only_block",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_bonded_mm_rescue_environment",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=MagicMock(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
    ):
        imp.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        minimize_bonded_mm_recovery(ctx, BondedMmMiniConfig(nstep_sd=0))
    ctx.unset.assert_called_once()
    ctx.reregister_mlpot.assert_called_once()


def test_measure_mm_bonded_strain_uses_mlpot_detached():
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    mock_py = MagicMock()
    saved_modules = {
        key: sys.modules.get(key)
        for key in (
            "pycharmm",
            "mmml.interfaces.pycharmmInterface.import_pycharmm",
        )
    }
    sys.modules["pycharmm"] = mock_py
    sys.modules["mmml.interfaces.pycharmmInterface.import_pycharmm"] = MagicMock()

    def fake_detached(detach_ctx, fn):
        detach_ctx.unset()
        try:
            return fn()
        finally:
            detach_ctx.reregister_mlpot()

    try:
        with patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_detached",
            side_effect=fake_detached,
        ) as detached, patch(
            "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_charmm_mm_block",
        ), patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
            return_value=0.5,
        ), patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_internal_energy_kcalmol",
            return_value=24.0,
        ), patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_bonded_term_kcalmol",
            return_value=24.0,
        ), patch(
            "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
        ):
            out = bonded_mm_recovery.measure_mm_bonded_strain_with_full_block(ctx)
    finally:
        for key, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod

    detached.assert_called_once()
    assert detached.call_args[0][0] is ctx
    ctx.unset.assert_called_once()
    ctx.reregister_mlpot.assert_called_once()
    assert out.grms_kcalmol_A == pytest.approx(0.5)
    assert out.angl_kcalmol == pytest.approx(24.0)


def test_with_mlpot_detached_unset_and_reregister():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _with_mlpot_detached
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    result = _with_mlpot_detached(ctx, lambda: 42)
    ctx.unset.assert_called_once()
    ctx.reregister_mlpot.assert_called_once()
    assert result == 42


def test_reregister_mlpot_reattaches_without_new_mlpot_or_nbond_rebuild():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    mlpot = MagicMock()
    ctx = MlpotContext(
        mlpot=mlpot,
        pyCModel=MagicMock(),
        params=None,
        model=None,
        ml_selection=MagicMock(),
        ml_Z=np.array([6, 1, 1, 1], dtype=int),
        use_pbc=True,
        cubic_box_side_A=31.0,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        return_value="all",
    ) as apply_block:
        ctx.reregister_mlpot()

    apply_block.assert_called_once_with(
        ctx.ml_selection, mm_internal_scale=0.0, verbose=False
    )
    mlpot.reattach_mlpot.assert_called_once_with()


def test_reregister_after_topology_reload_skips_upinb_rebuild():
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    positions = np.zeros((25, 3), dtype=float)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=positions,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
    ) as sync_pos, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.register_mlpot",
    ) as register, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.refresh_nbonds_after_mlpot_pbc",
    ) as refresh_pbc:
        bonded_mm_recovery._reregister_mlpot_after_topology_reload(ctx)

    ctx.reregister_mlpot.assert_called_once_with()
    sync_pos.assert_called_once_with(positions)
    register.assert_not_called()
    refresh_pbc.assert_not_called()


def test_assert_mlpot_user_active_reattaches_when_user_missing():
    import sys
    import types

    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        MlpotContext,
        assert_mlpot_user_active,
    )

    mlpot = MagicMock()
    mlpot.is_set = False
    ctx = MlpotContext(
        mlpot=mlpot,
        pyCModel=MagicMock(),
        params=None,
        model=None,
        ml_selection=MagicMock(),
        ml_Z=np.array([6, 1], dtype=int),
    )
    mock_py = types.ModuleType("pycharmm")
    mock_py.lingo = MagicMock()
    mock_settings = MagicMock()
    mock_settings.set_verbosity.return_value = 0
    mock_settings.set_warn_level.return_value = 0
    mock_settings.set_bomb_level.return_value = 0
    mock_py.settings = mock_settings
    mock_energy = types.ModuleType("pycharmm.energy")
    mock_energy.get_term_by_name = MagicMock()
    mock_energy.get_term_by_name.side_effect = [0.0, -123.4]
    with patch.dict(
        sys.modules,
        {
            "pycharmm": mock_py,
            "pycharmm.energy": mock_energy,
            "pycharmm.settings": mock_settings,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        return_value="all",
    ):
        user = assert_mlpot_user_active(ctx, context="test", quiet=True)

    ctx.mlpot.reattach_mlpot.assert_called_once()
    assert user == pytest.approx(-123.4)


def test_restore_workflow_nbonds_skips_nbond_rebuild():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext, restore_workflow_nbonds

    ctx = MlpotContext(
        mlpot=MagicMock(),
        pyCModel=MagicMock(),
        params=None,
        model=None,
        use_pbc=False,
    )
    mock_py = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._import_pycharmm",
        return_value=mock_py,
    ):
        restore_workflow_nbonds(ctx)

    mock_py.nbonds.update_bnbnd.assert_not_called()
    mock_py.nbonds.set_imgfrq.assert_not_called()
    mock_py.UpdateNonBondedScript.assert_not_called()


def test_bonded_mm_mini_watches_heat_even_when_after_is_mini_only():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        bonded_mm_mini_watches_stage,
    )

    args = argparse_namespace(bonded_mm_mini=True, bonded_mm_mini_after="mini")
    assert bonded_mm_mini_watches_stage(args, "heat") is True
    assert bonded_mm_mini_watches_stage(args, "mini") is True
    assert bonded_mm_mini_watches_stage(args, "equi") is False
    assert bonded_mm_mini_watches_stage(
        argparse_namespace(bonded_mm_mini=False, bonded_mm_mini_after="heat"),
        "heat",
    ) is False


def test_maybe_run_bonded_mm_mini_skips_when_grms_ok():
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        bonded_mm_grms_margin=0.0,
        quiet=True,
    )
    baseline = MmStrainBaseline(grms_kcalmol_A=12.0, internal_kcalmol=24.0, angl_kcalmol=24.0)
    current = MmStrainBaseline(grms_kcalmol_A=10.0, internal_kcalmol=24.0, angl_kcalmol=24.0)
    with patch.object(
        bonded_mm_recovery,
        "measure_mm_bonded_strain_with_full_block",
        return_value=current,
    ) as measure, patch.object(
        bonded_mm_recovery,
        "minimize_bonded_mm_recovery",
    ) as mini:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
        )
    measure.assert_called_once()
    mini.assert_not_called()
    assert ran is False


def test_maybe_run_bonded_mm_mini_always_runs_when_grms_ok():
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    ctx.pyCModel = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_always=True,
        bonded_mm_mini_after="heat",
        bonded_mm_mini_steps=25,
        bonded_mm_grms_margin=0.0,
        dyn_nprint=100,
        quiet=True,
        show_energy=False,
    )
    baseline = MmStrainBaseline(grms_kcalmol_A=12.0, internal_kcalmol=24.0, angl_kcalmol=24.0)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=False,
    ), patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=MmStrainBaseline(grms_kcalmol_A=0.1, internal_kcalmol=1.0, angl_kcalmol=1.0),
    ) as measure, patch.object(
        bonded_mm_recovery,
        "_run_hybrid_bonded_mlpot_recovery",
    ) as hybrid, patch.object(
        bonded_mm_recovery,
        "rewrite_dynamics_restart_validated",
        return_value=True,
    ) as rewrite:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
        )
    measure.assert_called_once()
    hybrid.assert_called_once()
    assert ran is True


def test_maybe_run_bonded_mm_mini_always_inplace_all_ml(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_always=True,
        bonded_mm_mini_after="heat",
        bonded_mm_mini_steps=40,
        dyn_nprint=100,
        quiet=True,
    )
    topology_psf = tmp_path / "cluster_for_vmd_benz_100.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=None,
            topology_psf=topology_psf,
        )
    inplace.assert_called_once()
    assert ran is True


def test_maybe_run_bonded_mm_mini_always_all_ml_mini_uses_inplace(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_always=True,
        bonded_mm_mini_after="mini",
        bonded_mm_mini_steps=25,
        dyn_nprint=50,
        quiet=True,
    )
    topology_psf = tmp_path / "cluster_for_vmd_dcm_100.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace, patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=MmStrainBaseline(grms_kcalmol_A=0.1, internal_kcalmol=1.0, angl_kcalmol=1.0),
    ):
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="mini",
            baseline=MmStrainBaseline(grms_kcalmol_A=0.5),
            topology_psf=topology_psf,
        )
    inplace.assert_called_once()
    assert ran is True


def test_recovery_reasons_angl_margin():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        MmStrainBaseline,
        _recovery_reasons,
    )

    base = MmStrainBaseline(grms_kcalmol_A=0.5, internal_kcalmol=24.0, angl_kcalmol=24.0)
    cur = MmStrainBaseline(grms_kcalmol_A=0.4, internal_kcalmol=25.0, angl_kcalmol=30.0)
    reasons = _recovery_reasons(
        cur, base, grms_margin=0.0, internal_margin=0.0, angl_margin=5.0
    )
    assert len(reasons) == 1
    assert "ANGL" in reasons[0]


def test_maybe_run_bonded_mm_mini_runs_when_grms_high():
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    ctx.pyCModel = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        bonded_mm_mini_steps=25,
        bonded_mm_grms_margin=0.0,
        dyn_nprint=100,
        quiet=True,
        show_energy=False,
    )
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    current = MmStrainBaseline(grms_kcalmol_A=20.0)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=False,
    ), patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=current,
    ), patch.object(
        bonded_mm_recovery,
        "_run_hybrid_bonded_mlpot_recovery",
    ) as hybrid:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
        )
    hybrid.assert_called_once()
    assert ran is True


def test_maybe_run_bonded_mm_mini_skips_heavy_when_heat_overlap(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock(n_monomers=90, use_pbc=False)
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        quiet=True,
    )
    topology_psf = tmp_path / "cluster.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch.object(
        bonded_mm_recovery,
        "_bonded_mm_skip_reason_after_heat_overlap",
        return_value="worst inter-monomer distance 0.71 Å < 0.50 Å",
    ) as skip, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            topology_psf=topology_psf,
        )
    skip.assert_called_once()
    inplace.assert_not_called()
    assert ran is False


def test_maybe_run_bonded_mm_mini_all_ml_skips_when_strain_ok(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="mini",
        quiet=True,
    )
    topology_psf = tmp_path / "cluster_for_vmd_dcm_100.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    ok = MmStrainBaseline(grms_kcalmol_A=0.45)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=ok,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="mini",
            baseline=baseline,
            topology_psf=topology_psf,
        )

    inplace.assert_not_called()
    assert ran is False


def test_maybe_run_bonded_mm_mini_all_ml_runs_inplace_when_strain_high(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="mini",
        quiet=True,
        bonded_mm_grms_margin=0.0,
    )
    topology_psf = tmp_path / "cluster_for_vmd_dcm_100.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    high = MmStrainBaseline(grms_kcalmol_A=20.0)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=high,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="mini",
            baseline=baseline,
            topology_psf=topology_psf,
        )

    inplace.assert_called_once()
    assert ran is True


def test_maybe_run_bonded_mm_mini_all_ml_uses_inplace_recovery(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        quiet=True,
        bonded_mm_grms_margin=0.0,
    )
    topology_psf = tmp_path / "cluster_for_vmd_dcm_10.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    high = MmStrainBaseline(grms_kcalmol_A=20.0)
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch.object(
        bonded_mm_recovery,
        "_measure_stage_bonded_strain",
        return_value=high,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_hybrid_bonded_mlpot_recovery",
    ) as inplace, patch.object(
        bonded_mm_recovery,
        "minimize_bonded_mm_recovery",
    ) as hybrid_mini:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
            topology_psf=topology_psf,
        )

    inplace.assert_called_once()
    hybrid_mini.assert_not_called()
    assert ran is True


def test_assert_pre_min_bonded_geometry_exits_on_high_angl():
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_max_angl_kcal=15.0,
        quiet=True,
    )
    mock_py = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_charmm_mm_block",
    ), patch.object(
        bonded_mm_recovery,
        "charmm_bonded_term_kcalmol",
        return_value=24.0,
    ), patch.object(
        bonded_mm_recovery,
        "charmm_internal_energy_kcalmol",
        return_value=24.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
    ), patch.dict(sys.modules, {"pycharmm": mock_py}), pytest.raises(SystemExit):
        bonded_mm_recovery.assert_pre_min_bonded_geometry(args)


def test_run_intra_overlap_rescue_all_ml_uses_bonded_sd_path(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_intra_monomer_overlap_rescue,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.ml_selection = MagicMock()
    ctx.ml_selection.get_atom_indexes.return_value = list(range(45))
    model = MagicMock()
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.0,
        intra_min_distance_A=1.0,
        n_monomers=9,
        use_pbc=True,
        rescue=OverlapRescueConfig(nstep_sd=50, verbose=False),
        pyCModel=model,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_all_ml_intra_overlap_rescue",
    ) as intra:
        run_intra_monomer_overlap_rescue(ctx, cfg)
    intra.assert_called_once()


def test_run_inter_overlap_rescue_all_ml_uses_bonded_vdw_path(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_inter_monomer_overlap_rescue,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock(spec=MlpotContext)
    ctx.ml_selection = MagicMock()
    ctx.ml_selection.get_atom_indexes.return_value = list(range(45))
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        n_monomers=9,
        use_pbc=False,
        topology_psf=topo,
        rescue=OverlapRescueConfig(nstep_sd=50, verbose=False),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_all_ml_inter_overlap_rescue",
    ) as inter:
        run_inter_monomer_overlap_rescue(ctx, cfg)
    inter.assert_called_once()


def test_all_ml_inter_overlap_rescue_uses_bonded_vdw_sd():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _run_all_ml_inter_overlap_rescue,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        n_monomers=200,
        use_pbc=True,
        pyCModel=MagicMock(),
        rescue=OverlapRescueConfig(nstep_sd=200, verbose=False),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.apply_charmm_position_noise",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_overlap_rescue",
    ) as bonded_rescue:
        _run_all_ml_inter_overlap_rescue(ctx, cfg)
    bonded_rescue.assert_called_once_with(ctx, cfg.rescue)


def test_run_inter_overlap_rescue_calls_bonded_vdw_rescue():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_inter_monomer_overlap_rescue,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        n_monomers=2,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=50, verbose=False),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_all_ml_inter_overlap_rescue",
    ) as inter:
        run_inter_monomer_overlap_rescue(ctx, cfg)
    inter.assert_called_once()


def test_run_intra_overlap_rescue_partial_ml_uses_light_path():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_intra_monomer_overlap_rescue,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.0,
        intra_min_distance_A=1.0,
        n_monomers=2,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=50, verbose=False),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_all_ml_intra_overlap_rescue",
    ) as intra:
        run_intra_monomer_overlap_rescue(ctx, cfg)
    intra.assert_called_once()


def test_rewrite_dynamics_restart_writes_current_state(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_from_current_state,
    )

    res = tmp_path / "equi.res"
    mock_py = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ):
        rewrite_dynamics_restart_from_current_state(res, write_unit=92)

    script = mock_py.lingo.charmm_script.call_args[0][0]
    assert str(res) in script
    assert "write restart unit 92" in script


def test_restore_charmm_state_from_restart_parses_and_syncs_positions(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    restart = tmp_path / "prior.res"
    restart.write_text(
        "REST     0     1\n"
        "       2 !NTITLE followed by title\n"
        "* test\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         2           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        " 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00\n"
        " 0.400000000000000D+00 0.500000000000000D+00 0.600000000000000D+00\n"
    )
    expected = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=float,
    )
    mock_py = MagicMock()
    synced: list[np.ndarray] = []

    def _capture_sync(pos):
        synced.append(np.asarray(pos, dtype=float))

    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._charmm_natom_count",
        return_value=2,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((2, 3), dtype=float),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        side_effect=_capture_sync,
    ):
        restore_charmm_state_from_restart(restart, read_unit=93)

    assert len(synced) == 1
    assert np.allclose(synced[0], expected)
    script = mock_py.lingo.charmm_script.call_args[0][0]
    assert "open read unit 93" in script
    assert str(restart) in script
    assert "read restart unit 93" in script
    assert "UPDATE" in script


def test_restore_charmm_state_from_restart_prefers_live_coords(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    restart = tmp_path / "prior.res"
    restart.write_text(
        "REST     0     1\n"
        "       2 !NTITLE followed by title\n"
        "* test\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         2           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        " 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00\n"
        " 0.400000000000000D+00 0.500000000000000D+00 0.600000000000000D+00\n"
    )
    live = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=float)
    synced: list[np.ndarray] = []

    def _capture_sync(pos):
        synced.append(np.asarray(pos, dtype=float))

    mock_py = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._charmm_natom_count",
        return_value=2,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=live.copy(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        side_effect=_capture_sync,
    ):
        restore_charmm_state_from_restart(restart, read_unit=93)

    assert len(synced) == 1
    assert np.allclose(synced[0], live)


def test_restore_charmm_state_from_restart_raises_natom_mismatch(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    restart = tmp_path / "stale.res"
    restart.write_text(
        "REST     0     1\n"
        "     125 !NTITLE followed by title\n"
        "* stale 25-mer\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "       125           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        + "\n".join(
            " 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00"
            for _ in range(125)
        )
        + "\n"
    )
    live = np.zeros((100, 3), dtype=float)
    mock_py = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._charmm_natom_count",
        return_value=100,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=live,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
    ) as mock_sync:
        with pytest.raises(RuntimeError, match="offline NATOM=125 vs CHARMM natom=100"):
            restore_charmm_state_from_restart(restart, read_unit=93)
    mock_sync.assert_not_called()


def test_restore_charmm_state_from_restart_raises_without_finite_coords(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    restart = tmp_path / "bad.res"
    restart.write_text(
        "REST     0     1\n"
        "       2 !NTITLE followed by title\n"
        "* test\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        "          NaN 0.200000000000000D+00 0.300000000000000D+00\n"
    )
    with pytest.raises(RuntimeError, match="no finite Cartesian coordinates"):
        restore_charmm_state_from_restart(restart)


def test_reload_pre_mlpot_topology_disabled_without_env_flag(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _reload_pre_mlpot_topology,
    )

    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock()
    with pytest.raises(RuntimeError, match="DELETE ATOM PSF reload is disabled"):
        _reload_pre_mlpot_topology(ctx, topology_psf=topo)


def test_reload_pre_mlpot_topology_uses_explicit_positions_not_charmm_array(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _reload_pre_mlpot_topology,
    )

    monkeypatch.setenv("MMML_ALLOW_PSF_DELETE_RELOAD", "1")

    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock()
    ctx.use_pbc = False
    ctx.charmm_cubic_box_side_A = None
    ctx.cubic_box_side_A = None
    explicit = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=float,
    )
    stale_nan = np.full((2, 3), np.nan)
    synced: list[np.ndarray] = []

    mock_py = MagicMock()
    mock_read = MagicMock()
    mock_py.read = mock_read

    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "pycharmm.read": mock_read,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=stale_nan,
    ) as get_pos, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        side_effect=lambda pos: synced.append(np.asarray(pos, dtype=float)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.setup_default_nbonds",
    ):
        _reload_pre_mlpot_topology(ctx, topology_psf=topo, positions=explicit)

    get_pos.assert_not_called()
    assert len(synced) == 1
    assert np.allclose(synced[0], explicit)
    mock_read.psf_card.assert_called_once_with(str(topo.resolve()))
    ctx.unset.assert_called_once()


def test_reload_pre_mlpot_topology_raises_on_nonfinite_explicit_positions(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _reload_pre_mlpot_topology,
    )

    monkeypatch.setenv("MMML_ALLOW_PSF_DELETE_RELOAD", "1")

    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock()
    bad = np.array([[1.0, np.nan, 3.0]], dtype=float)

    with pytest.raises(RuntimeError, match="pre-MLpot topology reload requires finite"):
        _reload_pre_mlpot_topology(ctx, topology_psf=topo, positions=bad)


def test_reload_pre_mlpot_topology_default_reads_finite_charmm_positions(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _reload_pre_mlpot_topology,
    )

    monkeypatch.setenv("MMML_ALLOW_PSF_DELETE_RELOAD", "1")

    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock()
    ctx.use_pbc = False
    ctx.charmm_cubic_box_side_A = None
    ctx.cubic_box_side_A = None
    from_charmm = np.array([[7.0, 8.0, 9.0]], dtype=float)
    synced: list[np.ndarray] = []

    mock_py = MagicMock()
    mock_read = MagicMock()
    mock_py.read = mock_read

    with patch.dict(
        "sys.modules",
        {
            "pycharmm": mock_py,
            "pycharmm.read": mock_read,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": MagicMock(),
        },
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=__import__("contextlib").nullcontext(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=from_charmm,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        side_effect=lambda pos: synced.append(np.asarray(pos, dtype=float)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.setup_default_nbonds",
    ):
        _reload_pre_mlpot_topology(ctx, topology_psf=topo)

    assert len(synced) == 1
    assert np.allclose(synced[0], from_charmm)


def test_run_extent_recovery_passes_restart_coords_to_all_ml_path(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_extent_recovery_from_prior_restart,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    restart = tmp_path / "prior.res"
    restart.write_text(
        "REST     0     1\n"
        "       2 !NTITLE followed by title\n"
        "* test\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         2           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        " 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00\n"
        " 0.400000000000000D+00 0.500000000000000D+00 0.600000000000000D+00\n"
    )
    expected = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=float,
    )
    topo = tmp_path / "cluster.psf"
    topo.write_text("psf", encoding="utf-8")
    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.0,
        n_monomers=1,
        use_pbc=False,
        topology_psf=topo,
        rescue=OverlapRescueConfig(nstep_sd=10, verbose=False),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.restore_geometry_from_ladder",
        return_value=restart,
    ) as restore, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.build_geometry_recovery_candidates",
        return_value=[],
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=expected,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery._run_all_ml_extent_recovery",
    ) as extent_path, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_bonded_mm_recovery",
    ) as light:
        run_extent_recovery_from_prior_restart(ctx, cfg, prior_restart=restart)

    restore.assert_called_once()
    assert restore.call_args[0][0][0] == restart.resolve()
    extent_path.assert_called_once()
    assert np.allclose(extent_path.call_args.kwargs["positions"], expected)
    light.assert_not_called()


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


def test_finalize_overlap_rescue_for_dynamics_reregisters_and_gates_grms():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        finalize_overlap_rescue_for_dynamics,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=4,
        rescue=OverlapRescueConfig(verbose=False),
        mlpot_rescue_mini_nstep=0,
        pyCModel=MagicMock(),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        return_value=12.0,
    ) as refresh, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=12.0,
    ):
        grms = finalize_overlap_rescue_for_dynamics(
            ctx, cfg, context="EQUI at step 2500"
        )
    ctx.reregister_mlpot.assert_called_once_with(verbose=False)
    assert refresh.call_count == 1
    assert grms == 12.0


def test_finalize_overlap_rescue_for_dynamics_aborts_on_high_grms():
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        finalize_overlap_rescue_for_dynamics,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=4,
        rescue=OverlapRescueConfig(verbose=False),
        mlpot_rescue_mini_nstep=0,
        pyCModel=MagicMock(),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        return_value=5000.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=5692.0,
    ):
        with pytest.raises(RuntimeError, match="post-overlap-rescue hybrid GRMS"):
            finalize_overlap_rescue_for_dynamics(
                ctx, cfg, context="EQUI at step 2500"
            )
