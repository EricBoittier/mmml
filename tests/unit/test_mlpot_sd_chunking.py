"""Chunked MLpot SD with periodic CHARMM UPDATE (inbfrq=0)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    MinimizeWithMlpotConfig,
    MlpotSdChunkResult,
    _effective_mlpot_sd_chunk_nstep,
    _maybe_abort_sd_on_grms,
    _maybe_abort_sd_on_grms_stall,
    _mlpot_sd_chunk_nstep,
    _resolved_sd_converged_grms,
    _run_minimize_in_chunks,
    invalidate_mlpot_calculator_caches,
    minimize_with_mlpot,
)


_DEFER_PATH_TEST_MARKERS = ("materialize", "prime_charmm")


@pytest.fixture(autouse=True)
def _disable_mlpot_mpi_defer_in_generic_chunk_tests(request):
    """Generic chunk tests use ``MagicMock(use_pbc=True)``; avoid real MPI defer path."""
    if any(marker in request.node.name for marker in _DEFER_PATH_TEST_MARKERS):
        yield
        return
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=False,
    ):
        yield


def test_mlpot_sd_chunk_nstep_smaller_for_pbc():
    ctx = MagicMock(use_pbc=True)
    cfg = MinimizeWithMlpotConfig(mlpot_ctx=ctx)
    assert _mlpot_sd_chunk_nstep(cfg) == 25
    cfg.mlpot_ctx = MagicMock(use_pbc=False)
    assert _mlpot_sd_chunk_nstep(cfg) == 500
    cfg.sd_chunk_nstep = 77
    assert _mlpot_sd_chunk_nstep(cfg) == 77


def test_effective_mlpot_sd_chunk_nstep_shrinks_when_grms_low():
    ctx = MagicMock(use_pbc=True)
    cfg = MinimizeWithMlpotConfig(mlpot_ctx=ctx)
    assert _effective_mlpot_sd_chunk_nstep(cfg, previous_grms=9.0) == 25
    assert _effective_mlpot_sd_chunk_nstep(cfg, previous_grms=3.0) == 25
    assert _effective_mlpot_sd_chunk_nstep(cfg, previous_grms=0.5) == 10
    cfg.sd_chunk_nstep = 8
    assert _effective_mlpot_sd_chunk_nstep(cfg, previous_grms=0.5) == 8


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


def test_maybe_abort_sd_on_grms_allows_transient_rise_after_calculator_mini():
    ctx = MagicMock()
    cfg = MinimizeWithMlpotConfig(
        mlpot_ctx=ctx,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_grms_watchdog_factor=2.5,
        verbose=False,
    )
    assert not _maybe_abort_sd_on_grms(
        cfg,
        initial_grms=0.5163,
        previous_grms=0.5163,
        current_grms=1.4881,
        pass_label="pass 1",
        step_label="after chunk 1",
    )
    assert _maybe_abort_sd_on_grms(
        cfg,
        initial_grms=0.5163,
        previous_grms=0.5163,
        current_grms=1200.0,
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
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=5.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=450,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is True
    assert minimize.run_sd.call_count == 4
    assert [call.kwargs["nstep"] for call in minimize.run_sd.call_args_list] == [200, 200, 25, 25]


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
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=12.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._rollback_mlpot_sd_chunk_geometry",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is False
    assert result.rolled_back is True
    assert result.last_grms == 12.0
    assert minimize.run_sd.call_count == 1


def test_run_minimize_in_chunks_watchdog_uses_sd_watchdog_initial_grms():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=600,
        mlpot_ctx=ctx,
        sd_chunk_nstep=200,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_grms_watchdog_factor=2.5,
        sd_watchdog_initial_grms=0.4,
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
        return_value=477.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._rollback_mlpot_sd_chunk_geometry",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is False
    assert result.rolled_back is True
    assert result.last_grms == 0.4
    assert minimize.run_sd.call_count == 1


def test_run_minimize_in_chunks_watchdog_rolls_back_after_chunk_blowup():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=600,
        mlpot_ctx=ctx,
        sd_chunk_nstep=200,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_converged_grms_kcalmol_A=5.0,
        sd_grms_watchdog_factor=2.5,
        verbose=True,
    )
    base_kw = {"inbfrq": 0, "ihbfrq": 0}
    good_positions = np.zeros((3, 3))

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        side_effect=[11.5, 1200.0],
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=12.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=good_positions,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._rollback_mlpot_sd_chunk_geometry",
    ) as rollback, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is False
    assert result.rolled_back is True
    assert result.last_grms == 11.5
    assert result.rolled_back_chunk == 1
    assert minimize.run_sd.call_count == 2
    rollback.assert_called_once()
    rollback_args = rollback.call_args
    assert rollback_args.args[0] is config
    np.testing.assert_array_equal(rollback_args.args[1], good_positions)
    assert rollback_args.kwargs == {
        "pass_label": "pass 1",
        "chunk_index": 1,
        "bad_grms": 1200.0,
        "good_grms": 11.5,
    }


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
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.materialize_deferred_mlpot_jax_before_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_mlpot_sd_then_abnr",
        return_value=MlpotSdChunkResult(completed=True),
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
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.materialize_deferred_mlpot_jax_before_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_mlpot_sd_then_abnr",
        return_value=MlpotSdChunkResult(completed=False),
    ):
        with pytest.raises(RuntimeError, match="watchdog"):
            minimize_with_mlpot(config)


def test_minimize_with_mlpot_continues_after_rollback():
    ctx = MagicMock()
    minimize = MagicMock()
    pycharmm = MagicMock()
    cons_fix = MagicMock()

    config = MinimizeWithMlpotConfig(
        nstep=3,
        nprint=10,
        verbose=True,
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
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.materialize_deferred_mlpot_jax_before_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_mlpot_sd_then_abnr",
        return_value=MlpotSdChunkResult(
            completed=False,
            rolled_back=True,
            last_grms=0.38,
            rolled_back_chunk=5,
        ),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ) as sync_lists, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
    ) as refresh_grms:
        minimize_with_mlpot(config)

    sync_lists.assert_called()
    refresh_grms.assert_called_once()


def test_maybe_abort_sd_on_grms_stall_detects_plateau():
    cfg = MinimizeWithMlpotConfig(
        mlpot_ctx=MagicMock(),
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_stall_patience_chunks=3,
        sd_stall_grms_abs_tol=0.1,
        verbose=False,
    )
    assert _maybe_abort_sd_on_grms_stall(
        cfg,
        previous_grms=272.82,
        current_grms=272.82,
        stagnant_chunks=3,
        pass_label="pass 1",
        step_label="after chunk 237",
    )
    assert not _maybe_abort_sd_on_grms_stall(
        cfg,
        previous_grms=272.82,
        current_grms=272.82,
        stagnant_chunks=2,
        pass_label="pass 1",
        step_label="after chunk 236",
    )


def test_run_minimize_in_chunks_stops_on_grms_plateau():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=600,
        mlpot_ctx=ctx,
        sd_chunk_nstep=25,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_abort_on_grms_increase=False,
        sd_stall_patience_chunks=3,
        sd_stall_grms_abs_tol=0.1,
        verbose=False,
    )
    base_kw = {"inbfrq": 0, "ihbfrq": 0}

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        return_value=272.8245,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=272.8245,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is False
    assert result.stalled is True
    assert result.last_grms == pytest.approx(272.8245)
    assert minimize.run_sd.call_count == 3


def test_run_minimize_in_chunks_exits_early_when_converged():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=600,
        mlpot_ctx=ctx,
        sd_chunk_nstep=200,
        pre_sd_bonded_recovery_grms_kcalmol_A=50.0,
        sd_abort_on_grms_increase=False,
        verbose=False,
    )
    base_kw = {"inbfrq": 0, "ihbfrq": 0}
    assert _resolved_sd_converged_grms(config) == pytest.approx(50.0)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        return_value=9.5,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=120.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ):
        result = _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=600,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    assert result.completed is True
    assert result.last_grms == pytest.approx(9.5)
    assert minimize.run_sd.call_count == 1


def test_run_minimize_in_chunks_materializes_deferred_jax_before_first_sd():
    ctx = MagicMock(use_pbc=True)
    minimize = MagicMock()
    pycharmm = MagicMock()
    config = MinimizeWithMlpotConfig(
        nstep=25,
        mlpot_ctx=ctx,
        sd_abort_on_grms_increase=False,
        verbose=False,
    )
    base_kw = {"inbfrq": 0, "ihbfrq": 0}

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._sync_mlpot_lists_after_sd_chunk",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_mlpot_sd_list_frequencies",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((1, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=1.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ) as recover_mpi, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.materialize_deferred_mlpot_jax_before_sd",
    ) as materialize:
        _run_minimize_in_chunks(
            minimize,
            pycharmm,
            config,
            base_kw,
            total_nstep=25,
            pass_label="pass 1",
            method="SD",
            run_attr="run_sd",
        )

    materialize.assert_called_once_with(
        ctx,
        verbose=False,
    )
    recover_mpi.assert_any_call(phase="immediately before MLpot SD steepd")


def test_materialize_deferred_mlpot_jax_before_sd_skips_without_mpi_defer():
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        materialize_deferred_mlpot_jax_before_sd,
    )

    ctx = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=False,
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is False


def test_materialize_deferred_mlpot_jax_before_sd_skips_update_sync_by_default():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    ctx = MagicMock(pyCModel=model, use_pbc=True, _mlpot_pre_sd_ener_probed=True)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ) as recover, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ) as sync_lists, patch.object(
        model,
        "get_pycharmm_calculator",
        return_value=MagicMock(
            spec=DecomposedMlpotCalculator,
            spherical_fn=MagicMock(),
        ),
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is True

    warmup.assert_called_once()
    assert recover.call_count == 2
    sync_lists.assert_not_called()


def test_materialize_deferred_mlpot_jax_before_sd_skips_charmm_ener_after_fresh_jax_materialize():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        _DeferredDecomposedMlpotCalculator,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._spherical_fn = None
    ctx = MagicMock(pyCModel=model, use_pbc=True, sd_watchdog_baseline_grms=None)
    calc = MagicMock(spec=DecomposedMlpotCalculator, spherical_fn=MagicMock())

    def _materialize_forces(*_args, **_kwargs):
        model._spherical_fn = MagicMock()
        return np.zeros((6, 3))

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.mlpot_spherical_forces_ev_angstrom",
        side_effect=_materialize_forces,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ) as prime, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ) as recover, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ) as sync_lists, patch.object(
        model,
        "get_pycharmm_calculator",
        side_effect=[
            _DeferredDecomposedMlpotCalculator(model),
            calc,
        ],
    ), patch.object(
        _DeferredDecomposedMlpotCalculator,
        "_ensure_real",
        return_value=calc,
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is True

    prime.assert_called_once()
    warmup.assert_called_once()
    assert recover.call_count == 3
    sync_lists.assert_not_called()


def test_materialize_deferred_mlpot_jax_before_sd_warms_callback_after_calculator_baseline():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._spherical_fn = object()
    ctx = MagicMock(
        pyCModel=model,
        use_pbc=True,
        sd_watchdog_baseline_grms=0.1585,
        _mlpot_pre_sd_ener_probed=True,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ) as recover, patch.object(
        model,
        "get_pycharmm_calculator",
        return_value=MagicMock(
            spec=DecomposedMlpotCalculator,
            spherical_fn=object(),
        ),
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is True

    warmup.assert_called_once()
    assert recover.call_count == 2


def test_materialize_deferred_mlpot_jax_before_sd_warms_callback_when_spherical_fn_ready():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._spherical_fn = object()
    ctx = MagicMock(
        pyCModel=model,
        use_pbc=True,
        sd_watchdog_baseline_grms=None,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ) as prime, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ) as recover, patch.object(
        model,
        "get_pycharmm_calculator",
        return_value=MagicMock(
            spec=DecomposedMlpotCalculator,
            spherical_fn=object(),
        ),
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is True

    prime.assert_called_once()
    warmup.assert_called_once()
    assert recover.call_count == 2


def test_materialize_deferred_mlpot_jax_before_sd_skips_probe_after_calculator_prep():
    test_materialize_deferred_mlpot_jax_before_sd_warms_callback_after_calculator_baseline()


def test_materialize_deferred_mlpot_jax_before_sd_skips_repeat_jax_on_second_call():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._spherical_fn = object()
    ctx = MagicMock(
        pyCModel=model,
        use_pbc=True,
        _mlpot_sd_jax_materialized=False,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ) as prime, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch.object(
        model,
        "get_pycharmm_calculator",
        return_value=MagicMock(
            spec=DecomposedMlpotCalculator,
            spherical_fn=object(),
        ),
    ):
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is True
        assert materialize_deferred_mlpot_jax_before_sd(ctx) is False

    assert warmup.call_count == 1
    assert prime.call_count == 2


def test_materialize_deferred_mlpot_jax_before_sd_skips_callback_warmup_when_model_flag_set():
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
        materialize_deferred_mlpot_jax_before_sd,
    )

    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._spherical_fn = object()
    model._pre_sd_callback_forward_warmup_done = True
    ctx = MagicMock(
        pyCModel=model,
        use_pbc=True,
        _mlpot_sd_jax_materialized=False,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.assert_mpi_launcher_for_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=np.zeros((6, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prime_charmm_hybrid_energy_before_mlpot_sd",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch.object(
        model,
        "get_pycharmm_calculator",
        return_value=MagicMock(
            spec=DecomposedMlpotCalculator,
            spherical_fn=object(),
        ),
    ):
        materialize_deferred_mlpot_jax_before_sd(ctx, verbose=True)

    warmup.assert_not_called()


def test_prime_charmm_hybrid_energy_before_mlpot_sd_under_mpirun():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prime_charmm_hybrid_energy_before_mlpot_sd,
    )

    ctx = MagicMock(_mlpot_pre_sd_ener_probed=False)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.ensure_ml_exclusions_before_mlpot_charmm_energy",
    ) as ensure_excl, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
        return_value=4.2,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._read_mlpot_user_energy_kcal",
        return_value=-12.5,
    ):
        grms = prime_charmm_hybrid_energy_before_mlpot_sd(ctx, verbose=False)

    ensure_excl.assert_called_once_with(
        ctx,
        context="Pre-MLpot SD ENER prime",
        force_rebuild=True,
    )
    assert grms == pytest.approx(4.2)
    assert ctx._mlpot_pre_sd_ener_probed is True


def test_prime_charmm_hybrid_energy_before_mlpot_sd_skips_serial():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prime_charmm_hybrid_energy_before_mlpot_sd,
    )

    ctx = MagicMock(_mlpot_pre_sd_ener_probed=False)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ) as probe:
        assert prime_charmm_hybrid_energy_before_mlpot_sd(ctx) is None

    probe.assert_not_called()


def test_prime_charmm_hybrid_energy_re_primes_when_stale_probed_flag_without_user():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        prime_charmm_hybrid_energy_before_mlpot_sd,
    )

    ctx = MagicMock(_mlpot_pre_sd_ener_probed=True)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.mlpot_skip_charmm_ener_force_before_first_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
        return_value=3.1,
    ) as probe, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._read_mlpot_user_energy_kcal",
        side_effect=[None, -8.0],
    ):
        grms = prime_charmm_hybrid_energy_before_mlpot_sd(ctx, verbose=False)

    assert grms == pytest.approx(3.1)
    probe.assert_called_once()
    assert ctx._mlpot_pre_sd_ener_probed is True
