"""Tests for bonded-MM recovery helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
    assert pbc_kw["inbfrq"] == -1
    assert "imgfrq" not in pbc_kw
    assert vac_kw["inbfrq"] == 0
    assert "imgfrq" not in vac_kw


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


def test_maybe_run_bonded_mm_mini_skips_when_grms_ok():
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_mm_recovery

    ctx = MagicMock()
    args = argparse_namespace(
        bonded_mm_mini=True,
        bonded_mm_mini_after="heat",
        bonded_mm_grms_margin=0.0,
        quiet=True,
    )
    baseline = MmStrainBaseline(grms_kcalmol_A=12.0, internal_kcalmol=24.0)
    with patch.object(
        bonded_mm_recovery,
        "measure_mm_grms_with_full_block",
        return_value=10.0,
    ) as measure, patch.object(
        bonded_mm_recovery,
        "minimize_bonded_mm_recovery",
    ) as mini, patch.object(
        bonded_mm_recovery,
        "rewrite_dynamics_restart_from_current_state",
    ) as resync:
        bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
        )
    measure.assert_called_once()
    mini.assert_not_called()
    resync.assert_called_once_with("/tmp/heat.res")


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
    with patch.object(
        bonded_mm_recovery,
        "measure_mm_grms_with_full_block",
        return_value=20.0,
    ), patch.object(
        bonded_mm_recovery,
        "minimize_bonded_mm_recovery",
    ) as mini, patch.object(
        bonded_mm_recovery,
        "rewrite_dynamics_restart_from_current_state",
    ) as resync:
        bonded_mm_recovery.maybe_run_bonded_mm_mini_after_stage(
            ctx,
            args,
            stage="heat",
            baseline=baseline,
            restart_path="/tmp/heat.res",
        )
    mini.assert_called_once()
    resync.assert_called_once_with("/tmp/heat.res")


def argparse_namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)
