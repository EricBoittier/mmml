"""Hybrid GRMS from calculator vs CHARMM."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    classify_hybrid_charmm_grms_mismatch,
    forces_grms_kcalmol_A,
    light_resync_mlpot_state,
    measure_hybrid_charmm_grms,
    mlpot_hybrid_grms_from_calculator,
    prepare_mlpot_hybrid_state_for_sd,
    probe_and_light_resync_if_desync,
    resolve_mlpot_grms_kcalmol_A,
)


def test_forces_grms_matches_rms_of_components():
    forces = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)
    # sqrt(mean(9, 0, 0, 0, 16, 0)) = sqrt(25/6)
    assert forces_grms_kcalmol_A(forces) == pytest.approx(float(np.sqrt(25.0 / 6.0)))


def test_mlpot_hybrid_grms_uses_spherical_fn():
    ctx = mock.Mock(use_pbc=True, cubic_box_side_A=50.0, pyCModel=mock.Mock())
    pos = np.zeros((2, 3), dtype=float)
    forces_ev = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=float)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_positions_angstrom",
        return_value=pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.mlpot_spherical_forces_ev_angstrom",
        return_value=forces_ev,
    ) as spherical:
        grms = mlpot_hybrid_grms_from_calculator(ctx, natom=2)

    spherical.assert_called_once()
    assert grms == pytest.approx(forces_grms_kcalmol_A(forces_ev * 23.060548867))


@pytest.mark.parametrize(
    ("hybrid", "charmm", "kind"),
    [
        (3.0, 2.0, "ok"),
        (0.5, 1.15, "ok"),
        (9.0, 1.15, "geometry_stress"),
        (7.0, 2.0, "geometry_stress"),
        (472.0, 1.15, "geometry_stress"),
        (30.0, 10.0, "desync_suspected"),
        (80.0, 6.0, "both_high"),
    ],
)
def test_classify_hybrid_charmm_grms_mismatch(hybrid, charmm, kind):
    assert classify_hybrid_charmm_grms_mismatch(hybrid, charmm) == kind


def test_resolve_mlpot_grms_reports_geometry_stress(capsys):
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.mlpot_hybrid_grms_from_calculator",
        return_value=80.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=2.5,
    ):
        grms = resolve_mlpot_grms_kcalmol_A(ctx, context="gate check")

    assert grms == pytest.approx(80.0)
    out = capsys.readouterr().out
    assert "hybrid GRMS=80.0000" in out
    assert "geometry stress" in out
    assert "possible desync" not in out


def test_resolve_mlpot_grms_reports_desync(capsys):
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.mlpot_hybrid_grms_from_calculator",
        return_value=30.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=10.0,
    ):
        grms = resolve_mlpot_grms_kcalmol_A(ctx, context="gate check")

    assert grms == pytest.approx(30.0)
    out = capsys.readouterr().out
    assert "possible desync" in out


def test_resolve_mlpot_grms_falls_back_to_charmm_without_ctx():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=12.34,
    ):
        grms = resolve_mlpot_grms_kcalmol_A(None, context="")
    assert grms == pytest.approx(12.34)


def test_probe_and_light_resync_if_desync_runs_resync():
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        return_value=mock.Mock(
            hybrid=30.0,
            charmm=10.0,
            ratio=3.0,
            kind="desync_suspected",
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.light_resync_mlpot_state",
        return_value=5.0,
    ) as resync:
        grms = probe_and_light_resync_if_desync(ctx, context="sync")

    resync.assert_called_once()
    assert grms == pytest.approx(5.0)


def test_probe_and_light_resync_skips_resync_for_geometry_stress():
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        return_value=mock.Mock(
            hybrid=9.0,
            charmm=1.15,
            ratio=7.8,
            kind="geometry_stress",
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.light_resync_mlpot_state",
    ) as resync:
        grms = probe_and_light_resync_if_desync(ctx, context="sync")

    resync.assert_not_called()
    assert grms == pytest.approx(9.0)


def test_probe_and_light_resync_skips_resync_when_hybrid_relaxed():
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        return_value=mock.Mock(
            hybrid=0.5,
            charmm=1.15,
            ratio=2.3,
            kind="ok",
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.light_resync_mlpot_state",
    ) as resync:
        grms = probe_and_light_resync_if_desync(ctx, context="sync")

    resync.assert_not_called()
    assert grms == pytest.approx(0.5)


def test_light_resync_reregisters_and_updates():
    ctx = mock.Mock(use_pbc=False)
    fake_pycharmm = mock.MagicMock()

    with mock.patch.dict(sys.modules, {"pycharmm": fake_pycharmm}), mock.patch(
        "mmml.interfaces.pycharmmInterface.import_pycharmm",
        create=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        return_value=mock.MagicMock(
            __enter__=mock.Mock(return_value=None),
            __exit__=mock.Mock(return_value=False),
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        return_value=mock.Mock(hybrid=3.0, charmm=2.5, ratio=1.2, kind="ok"),
    ):
        grms = light_resync_mlpot_state(ctx, context="resync")

    ctx.reregister_mlpot.assert_called_once()
    assert fake_pycharmm.lingo.charmm_script.call_args_list[0][0][0] == "ENER FORCE"
    assert fake_pycharmm.lingo.charmm_script.call_args_list[1][0][0] == "UPDATE"
    assert grms == pytest.approx(3.0)


def test_prepare_mlpot_hybrid_state_aborts_when_grms_stays_high():
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
        return_value=-1000.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        side_effect=[
            mock.Mock(hybrid=472.0, charmm=1.15, ratio=410.0, kind="geometry_stress"),
            mock.Mock(hybrid=470.0, charmm=1.1, ratio=427.0, kind="geometry_stress"),
        ],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_bonded_mm_recovery",
    ):
        with pytest.raises(RuntimeError, match="refusing MLpot SD"):
            prepare_mlpot_hybrid_state_for_sd(
                ctx,
                grms_limit=50.0,
                energy_limit=None,
                bonded_recovery_nstep=50,
                verbose=False,
                allow_high_grms=False,
            )


def test_prepare_mlpot_hybrid_state_resync_before_bonded_recovery():
    ctx = mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
        return_value=-100.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms_after_ener_force",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.measure_hybrid_charmm_grms",
        side_effect=[
            mock.Mock(hybrid=30.0, charmm=10.0, ratio=3.0, kind="desync_suspected"),
            mock.Mock(hybrid=40.0, charmm=8.0, ratio=5.0, kind="desync_suspected"),
            mock.Mock(hybrid=3.0, charmm=2.5, ratio=1.2, kind="ok"),
        ],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.light_resync_mlpot_state",
        return_value=40.0,
    ) as resync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_bonded_mm_recovery",
    ) as bonded:
        hybrid, user = prepare_mlpot_hybrid_state_for_sd(
            ctx,
            grms_limit=25.0,
            energy_limit=None,
            bonded_recovery_nstep=25,
            verbose=False,
        )

    resync.assert_called_once()
    bonded.assert_called_once()
    assert hybrid == pytest.approx(3.0)
    assert user == pytest.approx(-100.0)
