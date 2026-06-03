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
    assert pbc_kw["inbfrq"] == -1
    assert pbc_kw["ihbfrq"] == 50
    assert "imgfrq" not in pbc_kw
    assert vac_kw["inbfrq"] == -1
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

    with patch.object(block_terms, "_import_pycharmm") as imp:
        imp.return_value.lingo.charmm_script = MagicMock()
        block_terms.apply_bonded_mm_only_block()
    script = imp.return_value.lingo.charmm_script.call_args[0][0]
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

    with patch.object(block_terms, "_import_pycharmm") as imp:
        imp.return_value.lingo.charmm_script = MagicMock()
        block_terms.apply_bonded_vdw_recovery_block()
    script = imp.return_value.lingo.charmm_script.call_args[0][0]
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
                selection="TYPE CG321",
            )
        )

    script = pycharmm.lingo.charmm_script.call_args[0][0]
    assert "GEO sphere harm" in script
    assert "droff 10.000000 force 0.010000 outside" in script
    assert "sele TYPE CG321 end" in script
    assert "quartic" not in script.lower()


def test_apply_flat_bottom_workflow_accepts_selection():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    with patch.object(restraints, "center_cluster_at_origin") as center, patch.object(
        restraints,
        "setup_flat_bottom_sphere_mmfp",
    ) as setup:
        cfg = restraints.apply_flat_bottom_workflow(
            radius=10.0,
            force=0.01,
            selection="TYPE CG321",
        )

    center.assert_called_once()
    setup.assert_called_once()
    assert cfg is not None
    assert cfg.selection == "TYPE CG321"


def test_clear_mmfp_uses_block_command():
    from mmml.interfaces.pycharmmInterface.mlpot import restraints

    pycharmm = MagicMock()
    with patch.object(restraints, "_import_pycharmm", return_value=pycharmm):
        restraints.clear_mmfp_restraints()

    script = pycharmm.lingo.charmm_script.call_args[0][0]
    assert "MMFP\nCLEAR\nEND" in "\n".join(
        line.strip() for line in script.splitlines() if line.strip()
    )
    assert "MMFP CLEAR" not in script


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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=1.0,
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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=MagicMock(),
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

    apply_block.assert_called_once_with(ctx.ml_selection)
    mlpot.reattach_mlpot.assert_called_once_with()


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
        "measure_mm_bonded_strain_with_full_block",
        return_value=current,
    ), patch.object(
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
    mini.assert_called_once()
    assert ran is True


def test_maybe_run_bonded_mm_mini_all_ml_uses_heavy_reload(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        quiet=True,
    )
    topology_psf = tmp_path / "cluster_for_vmd_dcm_10.psf"
    topology_psf.write_text("* psf\n", encoding="utf-8")
    baseline = MmStrainBaseline(grms_kcalmol_A=5.0)
    result = bonded_mm_recovery.BondedMmRecoveryResult(
        ran_recovery=True,
        current=MmStrainBaseline(grms_kcalmol_A=20.0),
        reasons=("GRMS high",),
        heavy_reload=True,
    )
    with patch.object(
        bonded_mm_recovery,
        "_mlpot_covers_all_atoms",
        return_value=True,
    ), patch.object(
        bonded_mm_recovery,
        "_run_heavy_bonded_recovery_check",
        return_value=result,
    ) as heavy, patch.object(
        bonded_mm_recovery,
        "measure_mm_bonded_strain_with_full_block",
    ) as measure:
        ran = bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
            topology_psf=topology_psf,
        )

    heavy.assert_called_once_with(
        ctx,
        args,
        stage="heat",
        baseline=baseline,
        topology_psf=topology_psf,
    )
    measure.assert_not_called()
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
    ), patch.dict(sys.modules, {"pycharmm": mock_py}), pytest.raises(SystemExit):
        bonded_mm_recovery.assert_pre_min_bonded_geometry(args)


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


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)
