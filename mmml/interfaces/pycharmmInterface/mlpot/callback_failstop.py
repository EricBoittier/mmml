"""Stop CHARMM when the MLpot ctypes callback raises.

``calculate_charmm`` is registered as a ``CFUNCTYPE``. A Python exception in
that callback is printed and then turned into a 0.0 return; CHARMM keeps
``DYNA``. That is how a first-step "Molecule extent … exceeds" error became
7 ps of force-free motion at E=0.

This wrapper records the first unexpected exception and refuses later
callbacks. Production then ends the process with ``os._exit(1)`` (``MPI_Abort``
first when multi-rank), so the job cannot continue and its exit status is
nonzero. Tests (or ``MMML_MLPOT_CALLBACK_ABORT=raise``) raise
:class:`MlpotCallbackAborted` instead of exiting the process.

``stop`` (CHARMM ``STOP``) is opt-in only. CHARMM ``STOP`` ends the process with
exit status **0** and never returns to Python, so a failed run would look
successful to any wrapper that checks ``$?`` (verified 19 Sep 2026 on gpu08).

``_CallbackPairListUnavailable`` is handled inside ``calculate_charmm`` and
returns 0.0 during setup: ``assert_mlpot_user_active``'s recovery ladder needs
that. Once that check passes, it calls ``set_mlpot_dynamics_armed(True)`` and the
handler re-raises into this wrapper instead.
"""

from __future__ import annotations

import functools
import os
import sys
import traceback
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

ABORT_ENV = "MMML_MLPOT_CALLBACK_ABORT"


class MlpotCallbackAborted(RuntimeError):
    """Fatal MLpot callback failure. CHARMM must not continue with E=0."""


_first_error: BaseException | None = None
_dynamics_armed = False


def reset_mlpot_callback_failstop() -> None:
    """Clear the sticky fatal flag and the dynamics arm (unit tests)."""
    global _first_error, _dynamics_armed
    _first_error = None
    _dynamics_armed = False


def set_mlpot_dynamics_armed(armed: bool) -> None:
    """``True`` once USER is verified before dynamics: pair-list loss becomes fatal."""
    global _dynamics_armed
    _dynamics_armed = bool(armed)


def mlpot_dynamics_armed() -> bool:
    return _dynamics_armed


def mlpot_callback_failstop_error() -> BaseException | None:
    return _first_error


def resolve_mlpot_callback_abort_mode(name: str | None = None) -> str:
    """``raise`` | ``stop`` | ``exit``.

    Unset: ``raise`` while a pytest test is running (``PYTEST_CURRENT_TEST``), so
    a callback failure fails the test instead of ``os._exit``-killing the worker;
    ``exit`` otherwise. Not ``"pytest" in sys.modules``: CuPy's testing shim
    imports pytest, so every production run with the GPU pair list (#242) would
    pick ``raise``, and ctypes swallows that raise. ``stop`` is opt-in: CHARMM
    ``STOP`` exits with status 0 (see module docstring).
    """
    raw = (name if name is not None else os.environ.get(ABORT_ENV) or "").strip().lower()
    if raw in ("raise", "stop", "exit"):
        return raw
    if raw:
        raise ValueError(f"{ABORT_ENV} must be raise|stop|exit; got {raw!r}")
    return "raise" if os.environ.get("PYTEST_CURRENT_TEST") else "exit"


def abort_mlpot_callback(exc: BaseException, *, mode: str | None = None) -> None:
    """Print once, then raise, STOP CHARMM (exit status 0), or ``os._exit(1)``."""
    print(
        "FATAL: MLpot CHARMM callback failed; refusing to continue with E=0 / F=0. "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    chosen = resolve_mlpot_callback_abort_mode(mode)
    if chosen == "raise":
        raise MlpotCallbackAborted(str(exc)) from exc
    if chosen == "stop":
        try:
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            import pycharmm

            # Does not return: CHARMM STOP ends the process with exit status 0.
            pycharmm.lingo.charmm_script("STOP")
        except Exception as stop_exc:
            print(
                f"FATAL: CHARMM STOP after callback failure also failed ({stop_exc})",
                file=sys.stderr,
                flush=True,
            )
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    _mpi_abort_if_multirank(1)
    os._exit(1)


def _mpi_abort_if_multirank(code: int) -> None:
    """``os._exit`` on one rank leaves the others blocked in collectives."""
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is None:
        return
    try:
        if int(mpi.COMM_WORLD.Get_size()) > 1:
            mpi.COMM_WORLD.Abort(code)
    except Exception:
        pass


def failstop_calculate_charmm(fn: F) -> F:
    """Wrap ``calculate_charmm`` so an exception cannot become a 0.0 USER term."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        global _first_error
        if _first_error is not None:
            abort_mlpot_callback(_first_error)
        try:
            return fn(*args, **kwargs)
        except MlpotCallbackAborted:
            raise
        except Exception as exc:
            _first_error = exc
            abort_mlpot_callback(exc)

    return wrapped  # type: ignore[return-value]
