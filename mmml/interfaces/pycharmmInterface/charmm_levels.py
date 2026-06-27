"""CHARMM BOMBlev/WRNLev helpers (lazy PyCHARMM import for unit tests)."""

from __future__ import annotations

from contextlib import contextmanager


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


def run_charmm_script_quiet(script: str) -> None:
    """Run a CHARMM script at PRNLev/WRNLev 0; restore prior levels on exit."""
    import pycharmm

    old = _set_charmm_levels(prnlev=0, warnlev=0)
    try:
        pycharmm.lingo.charmm_script(script)
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_quiet_output():
    """Temporarily set PRNLev/WRNLev 0 (e.g. SD mini, overlap rescue)."""
    old = _set_charmm_levels(prnlev=0, warnlev=0)
    try:
        yield
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_silent_command(*, bomlev: int = -2):
    """Minimal console output with relaxed bomb level (ENER/UPDATE, USER checks)."""
    old = _set_charmm_levels(prnlev=0, warnlev=0, bomlev=int(bomlev))
    try:
        yield
    finally:
        _restore_charmm_levels(old)


@contextmanager
def charmm_relaxed_bomlev(level: int = -2):
    """Relax BOMBlev/WRNLev for RTF/PRM/PSF/CARD reads; restore on exit.

    Do not leave ``bomlev 0`` after parameter loads — benign read warnings would
    abort the job on the next CHARMM command (e.g. MLpot registration).
    """
    old = _set_charmm_levels(warnlev=int(level), bomlev=int(level))
    try:
        yield
    finally:
        _restore_charmm_levels(old)
