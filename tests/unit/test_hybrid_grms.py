"""Hybrid GRMS from calculator vs CHARMM."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    forces_grms_kcalmol_A,
    mlpot_hybrid_grms_from_calculator,
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


def test_resolve_mlpot_grms_prefers_calculator_and_warns_on_stale_charmm(capsys):
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
    assert "CHARMM GRMS=2.5000" in out
    assert "stale or MM-only" in out


def test_resolve_mlpot_grms_falls_back_to_charmm_without_ctx():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
        return_value=12.34,
    ):
        grms = resolve_mlpot_grms_kcalmol_A(None, context="")
    assert grms == pytest.approx(12.34)
