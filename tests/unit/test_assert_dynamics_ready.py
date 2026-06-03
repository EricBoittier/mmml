"""Pre-dynamics readiness checks (mocked CHARMM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import assert_dynamics_ready


def test_assert_dynamics_ready_rejects_zero_grms_without_user():
    energy = MagicMock()
    energy.get_grms.return_value = 0.0
    energy.get_term_by_name.return_value = 0.0

    with patch("pycharmm.lingo.charmm_script"), patch(
        "pycharmm.energy", energy, create=True
    ):
        with pytest.raises(RuntimeError, match="USER energy inactive"):
            assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True)


def test_assert_dynamics_ready_accepts_active_mlpot():
    energy = MagicMock()
    energy.get_grms.return_value = 0.12
    energy.get_term_by_name.return_value = -1007.0

    with patch("pycharmm.lingo.charmm_script"), patch(
        "pycharmm.energy", energy, create=True
    ):
        grms = assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True)
    assert grms == pytest.approx(0.12)
