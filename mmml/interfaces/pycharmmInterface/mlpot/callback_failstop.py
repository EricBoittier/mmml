"""Stop CHARMM when the MLpot ctypes callback raises.

``calculate_charmm`` is registered as a ``CFUNCTYPE``. A Python exception in
that callback is printed and then turned into a 0.0 return; CHARMM keeps
``DYNA``. That is how a first-step "Molecule extent … exceeds" error became
7 ps of force-free motion at E=0.

This wrapper records the first unexpected exception and refuses later
callbacks. Production then issues CHARMM ``STOP`` and ``os._exit(1)`` so the
job cannot continue. Tests (or ``MMML_MLPOT_CALLBACK_ABORT=raise``) raise
:class:`MlpotCallbackAborted` instead of exiting the process.
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


def reset_mlpot_callback_failstop() -> None:
    """Clear the sticky fatal flag (unit tests)."""
    global _first_error
    _first_error = None


def mlpot_callback_failstop_error() -> BaseException | None:
    return _first_error


def resolve_mlpot_callback_abort_mode(name: str | None = None) -> str:
    """``raise`` | ``stop`` | ``exit``.

    Unset: ``raise`` under pytest so a callback failure fails the test instead
    of ``os._exit``-killing the worker; ``stop`` otherwise.
    """
    raw = (name if name is not None else os.environ.get(ABORT_ENV) or "").strip().lower()
    if raw in ("raise", "stop", "exit"):
        return raw
    if raw:
        raise ValueError(f"{ABORT_ENV} must be raise|stop|exit; got {raw!r}")
    return "raise" if "pytest" in sys.modules else "stop"


def abort_mlpot_callback(exc: BaseException, *, mode: str | None = None) -> None:
    """Print once, then raise, STOP CHARMM, or ``os._exit(1)``."""
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

            # May not return if CHARMM unwinds from inside the callback.
            pycharmm.lingo.charmm_script("STOP")
        except Exception as stop_exc:
            print(
                f"FATAL: CHARMM STOP after callback failure also failed ({stop_exc})",
                file=sys.stderr,
                flush=True,
            )
    os._exit(1)


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
