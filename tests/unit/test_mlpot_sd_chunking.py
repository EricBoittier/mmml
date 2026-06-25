"""Chunked MLpot SD with periodic CHARMM UPDATE (inbfrq=0)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    MinimizeWithMlpotConfig,
    _maybe_abort_sd_on_grms,
    _mlpot_sd_chunk_nstep,
    _run_minimize_in_chunks,
    invalidate_mlpot_calculator_caches,
    minimize_with_mlpot,
)


def test_mlpot_sd_chunk_nstep_smaller_for_pbc():
    ctx = MagicMock(use_pbc=True)
    cfg = MinimizeWithMlpotConfig(mlpot_ctx=ctx)
    assert _mlpot_sd_chunk_nstep(cfg) == 50
    cfg.mlpot_ctx = MagicMock(use_pbc=False)
    assert _mlpot_sd_chunk_nstep(cfg) == 500
    cfg.sd_chunk_nstep = 77
    assert _mlpot_sd_chunk_nstep(cfg) == 77


def test_maybe_abort_sd_on_grms_respects_initial_cap():
    ctx = MagicMock()
    cfg = MinimizeWithMlpotConfig(
        mlpot_ctx=ctx,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_grms_watchdog_factor=2.5,
        verbose=True,
    )
    assert _maybe_abort_sd_on_grms(
        cfg,
        initial_grms=12.0,
        previous_grms=12.0,
        current_grms=31.0,
        pass_label="pass 1",
        step_label="after chunk 1",
    )
    assert not _maybe_abort_sd_on_grms(
        cfg,
        initial_grms=12.0,
        previous_grms=12.0,
        current_grms=29.0,
        pass_label="pass 1",
        step_label="after chunk 1",
    )


def test_maybe_abort_sd_on_grms_detects_chunk_to_chunk_rise():
    ctx = MagicMock()
    cfg = MinimizeWithMlpotConfig(
        mlpot_ctx=ctx,
        pre_sd_bonded_recovery_grms_kcalmol_A=None,
        sd_grms_watchdog_factor=2.0,
        verbose=False,
    )
    assert _maybe_abort_sd_on_grms(
        cfg,
        initial_grms=10.0,
        previous_grms=40.0,
        current_grms=85.0,
        pass_label="pass 1",
        step_label="after chunk 2",
    )


def test_invalidate_mlpot_calculator_caches_clears_update_fn():
    calc = MagicMock()
    calc._cached_update_fn = object()
    model = MagicMock()
    model.get_pycharmm_calculator.return_value = calc
    ctx = MagicMock(pyCModel=model)
    invalidate_mlpot_calculator_caches(ctx)
    assert calc._cached_update_fn is None


def test_run_minimize_in_chunks_splits_long_pbc_sd():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=450,
        nprint=10,
        mlpot_ctx=ctx,
        sd_chunk_nstep=200,
        sd_abort_on_grms_increase=False,
        verbose=False,
    )
    base_kw = {"tolenr": 1e-3, "tolgrd": 1e-3, "inbfrq": 0, "ihbfrq": 0}

    grms_values = iter([5.0, 4.0, 3.0])

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        side_effect=lambda *a, **k: next(grms_values),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=5.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        ok = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=450,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert ok is True
    assert minimize.run_sd.call_count == 3
    assert [call.kwargs["nstep"] for call in minimize.run_sd.call_args_list] == [200, 200, 50]


def test_run_minimize_in_chunks_watchdog_stops_early():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=600,
        mlpot_ctx=ctx,
        sd_chunk_nstep=200,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_grms_watchdog_factor=2.5,
        verbose=False,
    )
    base_kw = {"inbfrq": 0, "ihbfrq": 0}

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        return_value=200.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=12.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        ok = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert ok is False
    assert minimize.run_sd.call_count == 1


def test_minimize_with_mlpot_refreshes_grms_after_sync():
    ctx = MagicMock()
    minimize = MagicMock()
    pycharmm = MagicMock()
    cons_fix = MagicMock()

    config = MinimizeWithMlpotConfig(
        nstep=3,
        nprint=10,
        verbose=True,
        mlpot_ctx=ctx,
        sd_abort_on_grms_increase=False,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, cons_fix, MagicMock(), minimize, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ) as sync_lists, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_mlpot_sd_then_abnr",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.prepare_mlpot_hybrid_state_for_sd",
        return_value=(1.0, -10.0),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
    ) as refresh_grms:
        minimize_with_mlpot(config)

    sync_lists.assert_called()
    refresh_grms.assert_called_once()
    assert refresh_grms.call_args.kwargs.get("context") == "Post MLpot SD pass 1"


def test_minimize_with_mlpot_raises_when_sd_watchdog_aborts():
    ctx = MagicMock()
    minimize = MagicMock()
    pycharmm = MagicMock()
    cons_fix = MagicMock()

    config = MinimizeWithMlpotConfig(
        nstep=3,
        nprint=10,
        verbose=False,
        mlpot_ctx=ctx,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, cons_fix, MagicMock(), minimize, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.prepare_mlpot_hybrid_state_for_sd",
        return_value=(12.0, -100.0),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_mlpot_sd_then_abnr",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="watchdog"):
            minimize_with_mlpot(config)
