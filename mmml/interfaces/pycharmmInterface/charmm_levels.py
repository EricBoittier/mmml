"""CHARMM BOMBlev/WRNLev helpers (lazy PyCHARMM import for unit tests)."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Iterator


def _set_charmm_levels(
    *,
    prnlev: int | None = None,
    warnlev: int | None = None,
    bomlev: int | None = None,
) -> dict[str, int]:
    """Set CHARMM print/warning levels via the stream API (no ``CHARMM>`` echo)."""
    import pycharmm.settings as settings

    old: dict[str, int] = {}
    if prnlev is not None:
        old["prnlev"] = int(settings.set_verbosity(int(prnlev)))
    if warnlev is not None:
        old["warnlev"] = int(settings.set_warn_level(int(warnlev)))
    if bomlev is not None:
        old["bomlev"] = int(settings.set_bomb_level(int(bomlev)))
    return old


def _restore_charmm_levels(old: dict[str, int]) -> None:
    import pycharmm.settings as settings

    if "prnlev" in old:
        settings.set_verbosity(int(old["prnlev"]))
    if "warnlev" in old:
        settings.set_warn_level(int(old["warnlev"]))
    if "bomlev" in old:
        settings.set_bomb_level(int(old["bomlev"]))


@contextmanager
def _capture_fortran_stdio_pty() -> Iterator[str]:
    """Capture Fortran stdout through a pseudo-TTY (Linux MPI clusters)."""
    import pty
    import select
    import tempfile

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    master, slave = pty.openpty()
    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        prefix="mmml-charmm-pty-",
        suffix=".log",
    )
    tmp_path = tmp.name
    tmp.close()
    capture_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    os.dup2(slave, 1)
    os.dup2(slave, 2)
    os.close(slave)

    stop = threading.Event()

    def _drain_master() -> None:
        try:
            while True:
                if stop.is_set():
                    r, _, _ = select.select([master], [], [], 0.05)
                    if not r:
                        break
                else:
                    r, _, _ = select.select([master], [], [], 0.25)
                    if not r:
                        continue
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                try:
                    os.write(saved_out, data)
                except OSError:
                    pass
                try:
                    os.write(capture_fd, data)
                except OSError:
                    break
        except OSError:
            pass

    thread = threading.Thread(target=_drain_master, daemon=True)
    thread.start()
    try:
        yield tmp_path
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        stop.set()
        thread.join(timeout=10.0)
        try:
            while True:
                r, _, _ = select.select([master], [], [], 0.05)
                if not r:
                    break
                data = os.read(master, 65536)
                if not data:
                    break
                try:
                    os.write(saved_out, data)
                    os.write(capture_fd, data)
                except OSError:
                    break
        except OSError:
            pass
        os.close(master)
        os.close(capture_fd)
        os.close(saved_out)
        os.close(saved_err)


@contextmanager
def _capture_fortran_stdio_file() -> Iterator[str]:
    """Redirect Fortran stdout/stderr to a temp file (no live terminal copy)."""
    import tempfile

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        prefix="mmml-charmm-capture-",
        suffix=".log",
    )
    tmp_path = tmp.name
    tmp.close()
    log_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield tmp_path
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(log_fd)


@contextmanager
def capture_fortran_stdio() -> Iterator[str]:
    """Capture Fortran unit-6 output to a temp file (PTY on Linux when available).

    ``redirect_stdout`` only affects Python ``sys.stdout``; CHARMM Fortran writes
    directly to file descriptor 1.  The caller must read and unlink ``tmp_path``.
    """
    if sys.platform.startswith("linux"):
        try:
            import pty  # noqa: F401

            with _capture_fortran_stdio_pty() as tmp_path:
                yield tmp_path
            return
        except (ImportError, OSError):
            pass
    with _capture_fortran_stdio_file() as tmp_path:
        yield tmp_path


@contextmanager
def suppress_charmm_fortran_io():
    """Redirect CHARMM Fortran stdout/stderr to devnull (fd-level, not sys.stdout)."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def run_charmm_script_quiet(script: str) -> None:
    """Run a CHARMM script at PRNLev/WRNLev 0; restore prior levels on exit."""
    import pycharmm

    old = _set_charmm_levels(prnlev=0, warnlev=0)
    try:
        with suppress_charmm_fortran_io():
            pycharmm.lingo.charmm_script(script)
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_quiet_prnlev():
    """Mute PRNLev only; preserve warnlev/bomlev (needed for CGENFF PARRDR at -3)."""
    old = _set_charmm_levels(prnlev=0)
    try:
        yield
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_quiet_output():
    """Temporarily set PRNLev/WRNLev 0 and swallow Fortran stdout/stderr."""
    old = _set_charmm_levels(prnlev=0, warnlev=0)
    try:
        with suppress_charmm_fortran_io():
            yield
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_silent_command(*, bomlev: int = -2):
    """Minimal console output with relaxed bomb level (ENER/UPDATE, USER checks)."""
    old = _set_charmm_levels(prnlev=0, warnlev=0, bomlev=int(bomlev))
    try:
        with suppress_charmm_fortran_io():
            yield
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_relaxed_bomlev(level: int = -2):
    """Relax BOMBlev/WRNLev for RTF/PRM/PSF/CARD reads; restore on exit."""
    old = _set_charmm_levels(warnlev=int(level), bomlev=int(level))
    try:
        yield
    finally:
        _restore_charmm_levels(old)
