"""CHARMM BOMBlev/WRNLev helpers (lazy PyCHARMM import for unit tests)."""

from __future__ import annotations

from contextlib import contextmanager


def run_charmm_script_quiet(script: str) -> None:
    """Run a CHARMM script at PRNLev/WRNLev 0; restore prior levels on exit."""
    import pycharmm
    import pycharmm.settings as settings

    old_prn = settings.set_verbosity(0)
    old_wrn = settings.set_warn_level(0)
    pycharmm.lingo.charmm_script("PRNLev 0\nWRNLev 0")
    try:
        pycharmm.lingo.charmm_script(script)
    finally:
        settings.set_verbosity(old_prn)
        settings.set_warn_level(old_wrn)
        pycharmm.lingo.charmm_script(
            f"PRNLev {int(old_prn)}\nWRNLev {int(old_wrn)}"
        )


@contextmanager
def charmm_quiet_output():
    """Temporarily set PRNLev/WRNLev 0 (e.g. SD mini, overlap rescue)."""
    import pycharmm
    import pycharmm.settings as settings

    old_prn = settings.set_verbosity(0)
    old_wrn = settings.set_warn_level(0)
    pycharmm.lingo.charmm_script("PRNLev 0\nWRNLev 0")
    try:
        yield
    finally:
        settings.set_verbosity(old_prn)
        settings.set_warn_level(old_wrn)
        pycharmm.lingo.charmm_script(
            f"PRNLev {int(old_prn)}\nWRNLev {int(old_wrn)}"
        )


@contextmanager
def charmm_silent_command(*, bomlev: int = -2):
    """Minimal console output with relaxed bomb level (ENER/UPDATE, USER checks)."""
    import pycharmm
    import pycharmm.settings as settings

    old_prn = settings.set_verbosity(0)
    old_wrn = settings.set_warn_level(0)
    old_bl = settings.set_bomb_level(int(bomlev))
    pycharmm.lingo.charmm_script(f"PRNLev 0\nWRNLev 0\nbomlev {int(bomlev)}")
    try:
        yield
    finally:
        settings.set_verbosity(old_prn)
        settings.set_warn_level(old_wrn)
        settings.set_bomb_level(old_bl)
        pycharmm.lingo.charmm_script(
            f"PRNLev {int(old_prn)}\nWRNLev {int(old_wrn)}\nbomlev {int(old_bl)}"
        )


@contextmanager
def charmm_relaxed_bomlev(level: int = -2):
    """Relax BOMBlev/WRNLev for RTF/PRM/PSF/CARD reads; restore on exit.

    Do not leave ``bomlev 0`` after parameter loads — benign read warnings would
    abort the job on the next CHARMM command (e.g. MLpot registration).
    """
    import pycharmm
    import pycharmm.settings as settings

    old_bl = settings.set_bomb_level(int(level))
    old_wl = settings.set_warn_level(int(level))
    run_charmm_script_quiet(f"bomlev {int(level)}\nwrnlev {int(level)}")
    try:
        yield
    finally:
        settings.set_bomb_level(old_bl)
        settings.set_warn_level(old_wl)
        run_charmm_script_quiet(f"bomlev {int(old_bl)}\nwrnlev {int(old_wl)}")
