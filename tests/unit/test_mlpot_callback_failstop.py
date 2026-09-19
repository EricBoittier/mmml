"""A ctypes MLpot callback exception must stop CHARMM, never become a USER term."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot import callback_failstop
from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import (
    EXIT_CODE_ENV,
    MLPOT_CALLBACK_FAILURE_EXIT_CODE,
    MlpotCallbackAborted,
    fail_closed_callback,
    failstop_calculate_charmm,
    last_mlpot_callback_failure,
    mlpot_callback_failstop_error,
    mlpot_callback_failure_exit_code,
    mlpot_dynamics_armed,
    reset_mlpot_callback_failstop,
    set_mlpot_dynamics_armed,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _ExitCalled(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _raising_exit(code: int) -> None:
    raise _ExitCalled(code)


def test_exit_code_is_documented_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXIT_CODE_ENV, raising=False)
    assert MLPOT_CALLBACK_FAILURE_EXIT_CODE == 86
    assert mlpot_callback_failure_exit_code() == 86


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("91", 91), ("0", 86), ("-3", 86), ("300", 86), ("junk", 86), ("", 86)],
)
def test_exit_code_env_override_stays_nonzero(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    monkeypatch.setenv(EXIT_CODE_ENV, raw)
    assert mlpot_callback_failure_exit_code() == expected


def test_dynamics_arm_starts_off_and_reset_clears_it() -> None:
    assert mlpot_dynamics_armed() is False
    set_mlpot_dynamics_armed(True)
    assert mlpot_dynamics_armed() is True
    reset_mlpot_callback_failstop()
    assert mlpot_dynamics_armed() is False


def test_success_passes_return_value_through() -> None:
    calls: list[tuple] = []

    def cb(*args):
        calls.append(args)
        return 12.5

    guarded = fail_closed_callback(cb, exit_fn=_raising_exit)
    assert guarded(1, 2, 3) == 12.5
    assert calls == [(1, 2, 3)]


def test_exception_calls_exit_path_with_code_and_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(EXIT_CODE_ENV, raising=False)

    def cb(*_args):
        raise ValueError("Molecule extent 2.64 A exceeds the 2.60 A assumed")

    guarded = fail_closed_callback(cb, exit_fn=_raising_exit)
    with pytest.raises(_ExitCalled) as info:
        guarded(0)
    assert info.value.code == 86
    err = capsys.readouterr().err
    assert "MMML MLPOT CALLBACK FAILURE" in err
    assert "Traceback (most recent call last)" in err
    assert "Molecule extent 2.64 A" in err
    rec = last_mlpot_callback_failure()
    assert rec is not None
    assert rec["exc_type"] == "ValueError"
    assert rec["exit_code"] == 86


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(0), GeneratorExit()])
def test_base_exceptions_also_fail_closed(exc: BaseException) -> None:
    def cb(*_args):
        raise exc

    guarded = fail_closed_callback(cb, exit_fn=_raising_exit)
    with pytest.raises(_ExitCalled):
        guarded()


def test_returning_exit_fn_still_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """An injected exit_fn that returns must not let control go back to CHARMM."""
    seen: list[int] = []

    def fake_terminate(code: int):
        raise _ExitCalled(code)

    monkeypatch.setattr(callback_failstop, "terminate_after_callback_failure", fake_terminate)

    def cb(*_args):
        raise RuntimeError("boom")

    guarded = fail_closed_callback(cb, exit_fn=seen.append)
    with pytest.raises(_ExitCalled) as info:
        guarded()
    assert seen == [86]
    assert info.value.code == 86


def test_default_exit_path_uses_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_os_exit(code: int):
        raise _ExitCalled(code)

    monkeypatch.setattr(callback_failstop.os, "_exit", fake_os_exit)

    def cb(*_args):
        raise RuntimeError("boom")

    with pytest.raises(_ExitCalled) as info:
        fail_closed_callback(cb)()
    assert info.value.code == 86


def test_wrapping_is_idempotent() -> None:
    guarded = fail_closed_callback(lambda: 1.0)
    assert fail_closed_callback(guarded) is guarded


_CTYPES_CHILD = textwrap.dedent(
    """
    import atexit, ctypes, sys
    from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import fail_closed_callback

    atexit.register(lambda: print("ATEXIT-RAN", flush=True))
    FT = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int)
    calls = []

    def energy(step):
        calls.append(step)
        print(f"STEP {step}", flush=True)
        if step == 2:
            raise ValueError("incomplete COM-switched pair list")
        return 1.5

    mode = sys.argv[1]
    fn = FT(fail_closed_callback(energy) if mode == "guarded" else energy)
    # Stand-in for the Fortran integrator: it only sees the returned double.
    call = ctypes.cast(fn, FT)
    for step in range(5):
        e = call(step)
        print(f"RETURNED {step} {e}", flush=True)
    print("DYNAMICS-DONE", flush=True)
    """
)


def _run_ctypes_child(mode: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop(EXIT_CODE_ENV, None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.run(
        [sys.executable, "-c", _CTYPES_CHILD, mode],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_unguarded_ctypes_callback_fails_open() -> None:
    """Documents the bug: ctypes swallows the exception and the caller continues.

    The value handed back is whatever the return slot held (0.0 or a denormal
    such as 2.1e-314), never the energy.
    """
    proc = _run_ctypes_child("raw")
    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" in proc.stderr
    assert "RETURNED 2 " in proc.stdout
    assert "RETURNED 2 1.5" not in proc.stdout
    assert "STEP 3" in proc.stdout
    assert "DYNAMICS-DONE" in proc.stdout


def test_guarded_ctypes_callback_terminates_process() -> None:
    proc = _run_ctypes_child("guarded")
    assert proc.returncode == 86, (proc.stdout, proc.stderr)
    assert "STEP 2" in proc.stdout
    assert "RETURNED 2" not in proc.stdout
    assert "STEP 3" not in proc.stdout
    assert "DYNAMICS-DONE" not in proc.stdout
    # os._exit: atexit hooks (which could write success markers) never run.
    assert "ATEXIT-RAN" not in proc.stdout
    assert "Exception ignored" not in proc.stderr
    assert "incomplete COM-switched pair list" in proc.stderr
    assert "Traceback (most recent call last)" in proc.stderr


# --- method layer: sticky error, raise only -------------------------------


def test_method_wrapper_passes_through_energy():
    reset_mlpot_callback_failstop()

    @failstop_calculate_charmm
    def ok(*_args, **_kwargs):
        return -1717.77

    assert ok() == pytest.approx(-1717.77)
    assert mlpot_callback_failstop_error() is None


def test_method_wrapper_raises_and_sticks_without_exiting():
    reset_mlpot_callback_failstop()
    calls = {"n": 0}

    @failstop_calculate_charmm
    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise ValueError(
            "Molecule extent 2.100 A (max atom-to-centroid distance) exceeds the "
            "1.450 A assumed for the MM pair list radius"
        )

    with pytest.raises(MlpotCallbackAborted, match="Molecule extent") as info:
        boom()
    assert isinstance(info.value.__cause__, ValueError)
    assert calls["n"] == 1
    assert isinstance(mlpot_callback_failstop_error(), ValueError)

    # A later call must not run the body again.
    with pytest.raises(MlpotCallbackAborted, match="Molecule extent"):
        boom()
    assert calls["n"] == 1


def test_method_layer_error_through_entry_guard_exits_86(monkeypatch):
    monkeypatch.delenv(EXIT_CODE_ENV, raising=False)
    reset_mlpot_callback_failstop()

    @failstop_calculate_charmm
    def boom(*_args):
        raise ValueError("Molecule extent 3.0 A exceeds")

    guarded = fail_closed_callback(boom, exit_fn=_raising_exit)
    with pytest.raises(_ExitCalled) as info:
        guarded()
    assert info.value.code == 86
    assert last_mlpot_callback_failure()["exc_type"] == "MlpotCallbackAborted"


def test_rebind_registers_fail_closed_entry(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    bound = {}

    class _Calc:
        def calculate_charmm(self, *_args):
            raise ValueError("Molecule extent 3.0 A exceeds")

    class _Mlpot:
        func_type = staticmethod(lambda fn: fn)
        ml_Natoms = 1
        ml_indices = np.array([0])
        ml_Z = np.array([6])

        def unset_mlpot(self):
            return None

    fake_pycharmm = SimpleNamespace(
        lib=SimpleNamespace(
            charmm=SimpleNamespace(
                mlpot_set_func=lambda fn: bound.setdefault("fn", fn),
                mlpot_set_properties=lambda *a: None,
            )
        )
    )
    monkeypatch.setattr(mlpot_setup, "_import_pycharmm", lambda: fake_pycharmm)

    def fake_terminate(code: int):
        raise _ExitCalled(code)

    monkeypatch.setattr(callback_failstop, "terminate_after_callback_failure", fake_terminate)
    ctx = SimpleNamespace(
        pyCModel=SimpleNamespace(get_pycharmm_calculator=lambda: _Calc()),
        mlpot=_Mlpot(),
    )
    assert mlpot_setup.rebind_mlpot_calculator_from_pycmodel(ctx) is True
    assert getattr(bound["fn"], "__mmml_fail_closed__", False)
    with pytest.raises(_ExitCalled) as info:
        bound["fn"]()
    assert info.value.code == 86
