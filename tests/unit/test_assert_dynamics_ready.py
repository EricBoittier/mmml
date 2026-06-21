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
    scripts: list[str] = []
    fake_lingo.charmm_script = lambda script: scripts.append(str(script).strip().upper())

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


def test_assert_dynamics_ready_accepts_active_mlpot_low_grms(monkeypatch):
    _install_fake_pycharmm(monkeypatch, grms=0.0, user_kcal=-45817.0)

    grms = assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True)
    assert grms == pytest.approx(0.0)


def test_assert_dynamics_ready_calls_ener_force_when_mlpot_required(monkeypatch):
    refreshed: list[str] = []

    class _Ctx:
        def reregister_mlpot(self) -> None:
            refreshed.append("reregister")

    def _fake_refresh(ctx, *, context=""):
        refreshed.append(f"refresh:{context}")
        return 0.1

    fake_energy = types.ModuleType("pycharmm.energy")
    fake_energy.get_grms = lambda: 0.1
    fake_energy.get_term_by_name = lambda name: -100.0

    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda script: None

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.lingo = fake_lingo
    fake_pycharmm.energy = fake_energy

    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.energy", fake_energy)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        _fake_refresh,
    )

    ctx = _Ctx()
    assert_dynamics_ready(max_grms=50.0, require_mlpot_user=True, mlpot_ctx=ctx)
    assert refreshed == ["refresh:"]


def test_assert_dynamics_ready_retries_stale_mm_grms(monkeypatch):
    calls: list[str] = []

    class _Ctx:
        def reregister_mlpot(self) -> None:
            calls.append("reregister")

    grms_values = iter([175.0, 5.5])

    def _fake_refresh(ctx, *, context=""):
        calls.append(f"refresh:{context}")
        return next(grms_values)

    fake_energy = types.ModuleType("pycharmm.energy")
    fake_energy.get_grms = lambda: next(iter([5.5]))
    fake_energy.get_term_by_name = lambda name: -100.0

    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda script: None

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.lingo = fake_lingo
    fake_pycharmm.energy = fake_energy

    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.energy", fake_energy)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        _fake_refresh,
    )

    grms = assert_dynamics_ready(
        max_grms=50.0, require_mlpot_user=True, mlpot_ctx=_Ctx()
    )
    assert grms == pytest.approx(5.5)
    assert any("refresh:" in c for c in calls)
