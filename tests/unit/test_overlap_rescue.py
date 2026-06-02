"""Unit tests for MLpot overlap rescue minimization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_overlap_rescue
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import OverlapRescueConfig


def test_minimize_overlap_rescue_uses_vdw_block_and_restores_nbonds():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = MagicMock(spec=MlpotContext)
    ctx.use_pbc = True
    cfg = OverlapRescueConfig(nstep_sd=20, nstep_abnr=10, verbose=False)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._with_mlpot_block_restored",
        side_effect=lambda _ctx, fn: fn(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_bonded_vdw_recovery_block",
    ) as vdw_block, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.apply_recovery_nbonds",
    ) as rec_nb, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.restore_workflow_nbonds",
    ) as restore_nb, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=2.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=MagicMock(),
    ):
        imp.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        grms = minimize_overlap_rescue(ctx, cfg)

    vdw_block.assert_called_once()
    rec_nb.assert_called_once_with(ctx)
    restore_nb.assert_called_once_with(ctx)
    assert grms == 2.0
