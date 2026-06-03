"""Pre-dynamics readiness checks (mocked CHARMM)."""

from __future__ import annotations

import sys
import types

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import assert_dynamics_ready


def _install_fake_pycharmm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grms: float,
    user_kcal: float,
) -> None:
    """Stub pycharmm submodules without loading libcharmm (CI-safe)."""
    fake_energy = types.ModuleType("pycharmm.energy")
    fake_energy.get_grms = lambda: grms
    fake_energy.get_term_by_name = lambda name: user_kcal

    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda script: None

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.lingo = fake_lingo
    fake_pycharmm.energy = fake_energy

    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.energy", fake_energy)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)


def test_assert_dynamics_ready_rejects_zero_grms_without_user(monkeypatch):
    _install_fake_pycharmm(monkeypatch, grms=0.0, user_kcal=0.0)

    with pytest.raises(RuntimeError, match="USER energy inactive"):
        assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True)


def test_assert_dynamics_ready_accepts_active_mlpot(monkeypatch):
    _install_fake_pycharmm(monkeypatch, grms=0.12, user_kcal=-1007.0)

    grms = assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True)
    assert grms == pytest.approx(0.12)
