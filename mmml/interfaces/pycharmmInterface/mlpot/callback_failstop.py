"""Fail closed when the PyCHARMM MLpot energy callback raises.

CHARMM calls ``calculate_charmm`` through a ``ctypes.CFUNCTYPE`` pointer.
When a Python exception escapes a ctypes callback, ctypes prints
``Exception ignored on calling ctypes callback function`` and hands the
caller an undefined return value (0.0 or a stale denormal, never the energy).
CHARMM's Fortran then keeps integrating with that USER energy and whatever the
callback had written into ``dx/dy/dz`` so far, and the run's exit code,
restart files and "done" markers look like success. That is how a first-step
"Molecule extent ... exceeds" error became 7 ps of force-free NVE.

Two layers:

* :func:`failstop_calculate_charmm` decorates the ``calculate_charmm`` methods.
  It records the first error (sticky: later calls refuse to run the body) and
  raises :class:`MlpotCallbackAborted`. It never exits, so direct Python calls
  and unit tests see an ordinary exception.
* :func:`fail_closed_callback` wraps the ctypes entry point that is handed to
  ``mlpot_set_func`` (see :func:`install_fail_closed_energy_func`). On any
  ``BaseException`` it records the error, prints a banner and the full
  traceback to stderr (and to the original stderr when fd 2 is redirected,
  e.g. by ``suppress_charmm_fortran_io``), flushes Python, C stdio and
  gfortran unit buffers, and ends the process with
  :data:`MLPOT_CALLBACK_FAILURE_EXIT_CODE` (``86``) through ``os._exit``. No
  ``atexit`` hooks, ``finally`` blocks or context-manager exits run, so
  nothing can write a restart, DCD, stage summary, job manifest, ``next_run``
  advice or success line after the failure. Under a multi-rank MPI launch the
  communicator is aborted with the same code.

Why not CHARMM's own stop? ``STOP`` through ``lingo.charmm_script`` re-enters
CHARMM's command parser while the energy routine that called us is still on
the stack, and it ends in ``STOPCH``, whose Fortran ``STOP`` exits with status
0: a failed run would look successful. ``WRNDIE``/``DIE`` is not exported to
Python and ends with ``CALL EXIT(1)``, which cannot be told apart from any other
failure. The guard therefore terminates the process itself. It does not depend
on whether ``pytest`` is imported (CuPy imports ``pytest`` via ``cupy.testing``,
so production MD processes often have it in ``sys.modules``).

``MMML_MLPOT_CALLBACK_FAIL_EXIT_CODE`` (1..255) changes the exit code for tests
only; production runs should leave it unset.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import sys
import time
import traceback
from typing import Any, Callable, NoReturn, TypeVar

MLPOT_CALLBACK_FAILURE_EXIT_CODE = 86
"""Process exit status after an exception inside the MLpot energy callback."""

EXIT_CODE_ENV = "MMML_MLPOT_CALLBACK_FAIL_EXIT_CODE"
"""Test-only override of :data:`MLPOT_CALLBACK_FAILURE_EXIT_CODE`."""

_BANNER = "MMML MLPOT CALLBACK FAILURE"

_last_failure: dict[str, Any] | None = None


def _dup_or_none(fd: int) -> int | None:
    try:
        return os.dup(fd)
    except OSError:
        return None


# Copies of the process's stdout/stderr taken at import. CHARMM output is often
# silenced by pointing fd 1/2 at /dev/null (``suppress_charmm_fortran_io``) or
# a capture file; the failure report must still reach the real log.
_ORIG_STDOUT_FD = _dup_or_none(1)
_ORIG_STDERR_FD = _dup_or_none(2)


def _same_file(fd_a: int, fd_b: int | None) -> bool:
    if fd_b is None:
        return True
    try:
        a, b = os.fstat(fd_a), os.fstat(fd_b)
    except OSError:
        return True
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _emit(text: str, *, stream: Any, orig_stream: Any, fd: int, orig_fd: int | None) -> None:
    """Write to ``stream``; also to ``orig_fd`` when output was redirected away.

    Redirected means ``fd`` no longer refers to the file it did at import
    (fd-level silencing), or ``stream`` was swapped for another Python object
    (``contextlib.redirect_stderr``).
    """
    try:
        if stream is not None:
            stream.write(text)
            stream.flush()
    except Exception:
        pass
    redirected = stream is not orig_stream or not _same_file(fd, orig_fd)
    if orig_fd is not None and redirected:
        try:
            os.write(orig_fd, text.encode("utf-8", "replace"))
        except OSError:
            pass


def mlpot_callback_failure_exit_code() -> int:
    """Exit status used by the guard (``86`` unless the test override is set)."""
    raw = os.environ.get(EXIT_CODE_ENV, "").strip()
    if not raw:
        return MLPOT_CALLBACK_FAILURE_EXIT_CODE
    try:
        code = int(raw)
    except ValueError:
        return MLPOT_CALLBACK_FAILURE_EXIT_CODE
    # 0 would report success; keep the override inside the valid nonzero range.
    return code if 1 <= code <= 255 else MLPOT_CALLBACK_FAILURE_EXIT_CODE


def last_mlpot_callback_failure() -> dict[str, Any] | None:
    """The most recent failure recorded by the guard (``None`` if none)."""
    return None if _last_failure is None else dict(_last_failure)


def _flush_fortran_units() -> None:
    """Flush gfortran unit buffers (CHARMM writes its log through unit 6)."""
    name = ctypes.util.find_library("gfortran")
    if not name:
        return
    try:
        lib = ctypes.CDLL(name)
        flush_unit = lib._gfortran_flush_i4
    except (OSError, AttributeError):
        return
    for unit in (6, 0):
        try:
            flush_unit(ctypes.byref(ctypes.c_int(unit)))
        except Exception:
            pass


def flush_all_streams() -> None:
    """Best-effort flush of Python, C stdio and Fortran output buffers."""
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass
    try:
        _flush_fortran_units()
    except Exception:
        pass
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


def _mpi_abort_if_multirank(code: int) -> None:
    """Abort every rank when running under a multi-rank MPI launch."""
    mpi_mod = sys.modules.get("mpi4py.MPI")
    if mpi_mod is None:
        return
    try:
        comm = mpi_mod.COMM_WORLD
        if int(comm.Get_size()) > 1:
            comm.Abort(int(code))
    except Exception:
        pass


def terminate_after_callback_failure(code: int) -> NoReturn:
    """Default exit path: end the process without running any Python cleanup."""
    flush_all_streams()
    _mpi_abort_if_multirank(code)
    os._exit(int(code))


def _report(exc: BaseException, fn: Callable[..., Any], code: int) -> None:
    global _last_failure
    name = getattr(fn, "__qualname__", None) or repr(fn)
    _last_failure = {
        "callback": name,
        "exc_type": type(exc).__name__,
        "message": str(exc),
        "exit_code": int(code),
        "time": time.time(),
    }
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb = f"{type(exc).__name__}: {exc}\n"
    report = (
        f"\n{'=' * 72}\n{_BANNER}: {type(exc).__name__} in {name}\n"
        f"{exc}\n"
        "CHARMM cannot be stopped cleanly from inside the energy callback, and "
        "returning would let dynamics continue with a wrong USER energy and "
        f"stale forces. Terminating the process with exit code {code}; no "
        "restart, trajectory or stage marker is written after this point.\n"
        f"{'-' * 72}\n{tb}{'=' * 72}\n"
    )
    _emit(
        report,
        stream=sys.stderr if sys.stderr is not None else sys.__stderr__,
        orig_stream=sys.__stderr__,
        fd=2,
        orig_fd=_ORIG_STDERR_FD,
    )
    # Also mark stdout, which usually holds the CHARMM log (DYNA> lines).
    _emit(
        f"\n{_BANNER}: {type(exc).__name__}: {exc} (exit {code})\n",
        stream=sys.stdout if sys.stdout is not None else sys.__stdout__,
        orig_stream=sys.__stdout__,
        fd=1,
        orig_fd=_ORIG_STDOUT_FD,
    )


def fail_closed_callback(
    fn: Callable[..., Any],
    *,
    exit_fn: Callable[[int], Any] | None = None,
) -> Callable[..., Any]:
    """Wrap an MLpot ``calculate_charmm`` so any exception ends the process.

    ``exit_fn`` (default :func:`terminate_after_callback_failure`) receives the
    exit code; tests inject a function that raises instead. If an injected
    ``exit_fn`` returns, the process is still terminated with ``os._exit``:
    returning to CHARMM is never an option.
    """
    if getattr(fn, "__mmml_fail_closed__", False):
        return fn

    @functools.wraps(fn)
    def guarded(*args: Any) -> Any:
        try:
            return fn(*args)
        except BaseException as exc:  # noqa: BLE001 - every error must stop CHARMM
            code = mlpot_callback_failure_exit_code()
            # Flush first so CHARMM's buffered log (DYNA> lines) lands before
            # the failure banner in a merged stdout/stderr log.
            flush_all_streams()
            _report(exc, fn, code)
            flush_all_streams()
            if exit_fn is not None:
                exit_fn(code)
            terminate_after_callback_failure(code)

    guarded.__mmml_fail_closed__ = True  # type: ignore[attr-defined]
    return guarded


def install_fail_closed_energy_func(mlpot: Any, calc: Any | None = None, *, pycharmm_mod: Any | None = None) -> Any:
    """(Re)register ``mlpot``'s CHARMM callback through :func:`fail_closed_callback`.

    PyCHARMM's ``MLpot.__init__`` registers ``calculator.calculate_charmm``
    unguarded; call this right after constructing ``MLpot`` (and whenever the
    calculator is swapped) so every later energy call is fail-closed. Keeps the
    CFUNCTYPE object alive on ``mlpot`` (Fortran only holds the raw pointer).
    """
    if pycharmm_mod is None:
        import pycharmm as pycharmm_mod

    if calc is None:
        calc = mlpot.calculator
    energy_func = mlpot.func_type(fail_closed_callback(calc.calculate_charmm))
    mlpot.calculator = calc
    mlpot.energy_func = energy_func
    mlpot._energy_func_keepalive = (calc, energy_func)
    pycharmm_mod.lib.charmm.mlpot_set_func(energy_func)
    return energy_func


F = TypeVar("F", bound=Callable[..., Any])


class MlpotCallbackAborted(RuntimeError):
    """Fatal MLpot callback failure. CHARMM must not continue with this energy."""


_first_error: BaseException | None = None


def reset_mlpot_callback_failstop() -> None:
    """Clear the sticky fatal flag (unit tests)."""
    global _first_error
    _first_error = None


def mlpot_callback_failstop_error() -> BaseException | None:
    return _first_error


def abort_mlpot_callback(exc: BaseException) -> NoReturn:
    """Raise :class:`MlpotCallbackAborted` for ``exc``.

    Only raises: stopping the process is the job of the ctypes entry guard
    (:func:`fail_closed_callback`), which catches this exception.
    """
    if isinstance(exc, MlpotCallbackAborted):
        raise exc
    raise MlpotCallbackAborted(f"{type(exc).__name__}: {exc}") from exc


def failstop_calculate_charmm(fn: F) -> F:
    """Make ``calculate_charmm`` refuse to run again after its first error."""

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


__all__ = [
    "EXIT_CODE_ENV",
    "MLPOT_CALLBACK_FAILURE_EXIT_CODE",
    "MlpotCallbackAborted",
    "abort_mlpot_callback",
    "failstop_calculate_charmm",
    "mlpot_callback_failstop_error",
    "reset_mlpot_callback_failstop",
    "fail_closed_callback",
    "flush_all_streams",
    "install_fail_closed_energy_func",
    "last_mlpot_callback_failure",
    "mlpot_callback_failure_exit_code",
    "terminate_after_callback_failure",
]
