"""CHARMM BOMBlev/WRNLev helpers (lazy PyCHARMM import for unit tests)."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _set_charmm_levels(
    *,
    prnlev: int | None = None,
    warnlev: int | None = None,
    bomlev: int | None = None,
) -> dict[str, int]:
    """Set CHARMM print/warning/bomb levels via the stream API (no ``CHARMM>`` echo)."""
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
def tee_fortran_stdio() -> Iterator[str]:
    """Duplicate Fortran stdout/stderr to the terminal and a temp capture file."""
    import select
    import tempfile

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    read_fd, write_fd = os.pipe()
    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        prefix="mmml-charmm-tee-",
        suffix=".log",
    )
    tmp_path = tmp.name
    tmp.close()
    capture_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    stop = threading.Event()

    def _forward() -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            while True:
                if stop.is_set():
                    r, _, _ = select.select([reader], [], [], 0.05)
                    if not r:
                        break
                data = reader.read(65536)
                if not data:
                    break
                os.write(saved_out, data)
                os.write(capture_fd, data)

    thread = threading.Thread(target=_forward, daemon=True)
    thread.start()
    try:
        yield tmp_path
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        stop.set()
        thread.join(timeout=10.0)
        os.close(capture_fd)


@contextmanager
def capture_fortran_stdio(*, tee: bool = False) -> Iterator[str]:
    """Redirect (or tee) OS fds 1/2 so Fortran unit-6 output is captured to a temp file.

    ``redirect_stdout`` only affects Python ``sys.stdout``; CHARMM Fortran writes
    directly to file descriptor 1.  The caller must read and unlink ``tmp_path``.
    """
    if tee:
        with tee_fortran_stdio() as tmp_path:
            yield tmp_path
        return

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
    """Temporarily set PRNLev/WRNLev 0 and swallow Fortran stdout/stderr.

    Do not nest inside :func:`charmm_relaxed_bomlev` for CGENFF ``READ PARAM`` —
    WRNLev 0 prints PARRDR level -3 ``Null nonbond group`` banners. Prefer
    :func:`charmm_quiet_prnlev` plus :func:`suppress_charmm_fortran_io` there.
    """
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
    """Relax BOMBlev/WRNLev for RTF/PRM/PSF/CARD reads; restore on exit.

    Use ``level=-5`` for ``READ PARAM APPEND`` (CGENFF zeroed/full swaps) so PARMIO
    / PARRDR level -3 nonbond rebuild warnings do not abort at default ``-2``.

    Do not leave ``bomlev 0`` after parameter loads — benign read warnings would
    abort the job on the next CHARMM command (e.g. MLpot registration).
    """
    old = _set_charmm_levels(warnlev=int(level), bomlev=int(level))
    try:
        yield
    finally:
        _restore_charmm_levels(old)
