"""Unit tests for MLpot overlap rescue minimization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_overlap_rescue
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import OverlapRescueConfig


def test_apply_recovery_nbonds_skips_all_ml_pbc_upinb():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext, apply_recovery_nbonds

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = True
    ctx.ml_selection.get_atom_indexes.return_value = list(range(450))
    pycharmm = MagicMock()
    pycharmm.coor.get_natom.return_value = 450

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._import_pycharmm",
        return_value=pycharmm,
    ):
        apply_recovery_nbonds(ctx)

    pycharmm.nbonds.update_bnbnd.assert_not_called()
    pycharmm.UpdateNonBondedScript.assert_not_called()


def test_minimize_overlap_rescue_uses_vdw_block_and_restores_nbonds():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = True
    cfg = OverlapRescueConfig(nstep_sd=20, nstep_abnr=10, verbose=False)
    pycharmm = MagicMock()
    minimize = MagicMock()

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_detached",
        side_effect=lambda _ctx, fn: fn(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.apply_recovery_nbonds",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_rescue_lists",
    ) as prep_lists, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_bonded_vdw_recovery_block",
    ) as vdw_block, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.restore_workflow_nbonds",
    ) as restore_nb, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, MagicMock(), MagicMock(), minimize),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=2.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=MagicMock(),
    ):
        grms = minimize_overlap_rescue(ctx, cfg)

    vdw_block.assert_called_once()
    prep_lists.assert_called_once_with(ctx)
    restore_nb.assert_called_once_with(ctx)
    minimize.run_sd.assert_called_once()
    minimize.run_abnr.assert_called_once()
    ener_calls = [
        c.args[0]
        for c in pycharmm.lingo.charmm_script.call_args_list
        if c.args
    ]
    assert ener_calls == ["ENER"]
    assert grms == 2.0
