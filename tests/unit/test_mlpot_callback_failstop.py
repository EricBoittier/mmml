"""A ctypes MLpot callback exception must not become E=0 while CHARMM continues."""

from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import (
    ABORT_ENV,
    MlpotCallbackAborted,
    abort_mlpot_callback,
    failstop_calculate_charmm,
    mlpot_callback_failstop_error,
    reset_mlpot_callback_failstop,
    resolve_mlpot_callback_abort_mode,
)


def test_abort_mode_raise_under_pytest_by_default():
    assert resolve_mlpot_callback_abort_mode("") == "raise"
    assert resolve_mlpot_callback_abort_mode("stop") == "stop"
    assert resolve_mlpot_callback_abort_mode("exit") == "exit"
    with pytest.raises(ValueError, match="raise|stop|exit"):
        resolve_mlpot_callback_abort_mode("continue")


def test_wrapper_passes_through_energy():
    reset_mlpot_callback_failstop()

    @failstop_calculate_charmm
    def ok(*_args, **_kwargs):
        return -1717.77

    assert ok() == pytest.approx(-1717.77)
    assert mlpot_callback_failstop_error() is None


def test_wrapper_aborts_on_molecule_extent_and_sticks(monkeypatch):
    monkeypatch.setenv(ABORT_ENV, "raise")
    reset_mlpot_callback_failstop()
    calls = {"n": 0}

    @failstop_calculate_charmm
    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise ValueError(
            "Molecule extent 2.100 A (max atom-to-centroid distance) exceeds the "
            "1.450 A assumed for the MM pair list radius"
        )

    with pytest.raises(MlpotCallbackAborted, match="Molecule extent"):
        boom()
    assert calls["n"] == 1
    assert isinstance(mlpot_callback_failstop_error(), ValueError)

    # A later callback must not run the body (that was the 7 ps of E=0 dynamics).
    with pytest.raises(MlpotCallbackAborted, match="Molecule extent"):
        boom()
    assert calls["n"] == 1


def test_rebind_registers_failstop_wrapper(monkeypatch):
    from types import SimpleNamespace

    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    bound = {}

    class _Calc:
        def calculate_charmm(self, *_args, **_kwargs):
            raise ValueError("Molecule extent 3.0 A exceeds")

    class _Mlpot:
        func_type = staticmethod(lambda fn: fn)
        ml_Natoms = 1
        ml_indices = __import__("numpy").array([0])
        ml_Z = __import__("numpy").array([6])

        def unset_mlpot(self):
            return None

    fake_lib = SimpleNamespace(
        charmm=SimpleNamespace(
            mlpot_set_func=lambda fn: bound.setdefault("fn", fn),
            mlpot_set_properties=lambda *a: None,
        )
    )
    fake_pycharmm = SimpleNamespace(lib=fake_lib)
    monkeypatch.setattr(mlpot_setup, "_import_pycharmm", lambda: fake_pycharmm)
    monkeypatch.setenv(ABORT_ENV, "raise")
    reset_mlpot_callback_failstop()

    ctx = SimpleNamespace(
        pyCModel=SimpleNamespace(get_pycharmm_calculator=lambda: _Calc()),
        mlpot=_Mlpot(),
    )
    assert mlpot_setup.rebind_mlpot_calculator_from_pycmodel(ctx) is True
    with pytest.raises(MlpotCallbackAborted, match="Molecule extent"):
        bound["fn"]()


def test_abort_stop_falls_through_to_exit_when_charmm_stop_fails(monkeypatch):
    monkeypatch.setenv(ABORT_ENV, "stop")
    exited = {}

    def _fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.callback_failstop.os._exit",
        _fake_exit,
    )
    with pytest.raises(SystemExit) as ei:
        abort_mlpot_callback(ValueError("extent"), mode="stop")
    assert ei.value.code == 1
    assert exited["code"] == 1
