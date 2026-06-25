"""Unit tests for monomer cons_fix CLI and minimization wiring (no CHARMM runtime)."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    format_resid_constraint_message,
    parse_resid_list,
    resolve_constrain_resids,
    resolve_fix_resids,
    setup_cons_fix_for_resids,
    turn_off_cons_fix,
    validate_resids_for_cluster,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    MinimizeWithMlpotConfig,
    minimize_with_mlpot,
)


def test_parse_resid_list_dedupes_and_orders():
    assert parse_resid_list("3,1,3,2") == [3, 1, 2]


def test_parse_resid_list_rejects_invalid():
    with pytest.raises(ValueError, match=">= 1"):
        parse_resid_list("0,1")


def test_resolve_fix_resids_no_fix_flag():
    args = argparse.Namespace(no_fix=True, fix_resid=None, fix_resids="1,2")
    assert resolve_fix_resids(args) == []


def test_resolve_fix_resids_from_string():
    args = argparse.Namespace(no_fix=False, fix_resid=None, fix_resids="1,3")
    assert resolve_fix_resids(args) == [1, 3]


def test_resolve_fix_resids_deprecated_single():
    args = argparse.Namespace(no_fix=False, fix_resid=2, fix_resids="")
    assert resolve_fix_resids(args) == [2]


def test_resolve_constrain_resids_empty():
    args = argparse.Namespace(constrain_resids="")
    assert resolve_constrain_resids(args) == []


def test_validate_resids_for_cluster_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        validate_resids_for_cluster([4], 3)


def test_format_resid_constraint_message_none():
    assert "no monomers constrained" in format_resid_constraint_message(
        [], context="cons_fix"
    )


def _patch_pycharmm_cons_fix(cons_fix: MagicMock):
    fake_pycharmm = MagicMock()
    fake_pycharmm.cons_fix = cons_fix
    return patch.dict(
        sys.modules,
        {"pycharmm": fake_pycharmm, "pycharmm.cons_fix": cons_fix},
    )


def test_setup_cons_fix_for_resids_calls_pycharmm():
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [1, 2, 3]
    cons_fix = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.select_by_resids",
        return_value=fake_sel,
    ), _patch_pycharmm_cons_fix(cons_fix):
        sel = setup_cons_fix_for_resids([1])
    cons_fix.setup.assert_called_once_with(fake_sel)
    assert sel is fake_sel


def test_setup_cons_fix_raises_when_selection_empty():
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = []
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.select_by_resids",
        return_value=fake_sel,
    ), _patch_pycharmm_cons_fix(MagicMock()):
        with pytest.raises(RuntimeError, match="no atoms"):
            setup_cons_fix_for_resids([99])


def test_turn_off_cons_fix():
    cons_fix = MagicMock()
    with _patch_pycharmm_cons_fix(cons_fix):
        turn_off_cons_fix()
    cons_fix.turn_off.assert_called_once()


def test_prepare_mlpot_sd_list_frequencies_clears_imgfrq_when_inbfrq_zero():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _prepare_mlpot_sd_list_frequencies,
    )

    pycharmm = MagicMock()
    _prepare_mlpot_sd_list_frequencies(pycharmm, sd_kw={"inbfrq": 0, "ihbfrq": 0})
    pycharmm.nbonds.set_imgfrq.assert_called_once_with(0)
    pycharmm.nbonds.set_inbfrq.assert_called_once_with(0)
    pycharmm.reset_mock()
    _prepare_mlpot_sd_list_frequencies(pycharmm, sd_kw={"inbfrq": -1})
    pycharmm.nbonds.set_imgfrq.assert_not_called()


def test_minimize_with_mlpot_runs_two_sd_passes_when_fixed_selection_set():
    fixed_sel = MagicMock()
    fixed_sel.get_atom_indexes.return_value = list(range(10))
    cons_fix = MagicMock()
    minimize = MagicMock()
    pycharmm = MagicMock()

    config = MinimizeWithMlpotConfig(
        fixed_ml_selection=fixed_sel,
        nstep=5,
        nprint=10,
        verbose=False,
        show_energy=False,
        save=False,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, cons_fix, MagicMock(), minimize, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ) as mock_domdec, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ):
        assert minimize_with_mlpot(config) is True

    mock_domdec.assert_called_once_with(context="MLpot SD minimize")

    assert minimize.run_sd.call_count == 2
    assert pycharmm.nbonds.set_imgfrq.call_count == 4
    assert pycharmm.nbonds.set_inbfrq.call_count == 4
    cons_fix.setup.assert_called_once_with(fixed_sel)
    cons_fix.turn_off.assert_called_once()


def test_minimize_with_mlpot_asserts_user_when_ctx_provided():
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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.prepare_mlpot_hybrid_state_for_sd",
        return_value=(0.0, 0.0),
    ) as prepare_sd:
        minimize_with_mlpot(config)

    prepare_sd.assert_called_once()


def test_minimize_with_mlpot_single_pass_when_no_fix():
    minimize = MagicMock()
    pycharmm = MagicMock()
    cons_fix = MagicMock()

    config = MinimizeWithMlpotConfig(
        fixed_ml_selection=None,
        nstep=3,
        nprint=10,
        verbose=False,
        show_energy=False,
        save=False,
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
    ):
        minimize_with_mlpot(config)

    assert minimize.run_sd.call_count == 1
    cons_fix.setup.assert_not_called()
