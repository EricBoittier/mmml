"""CHARMM BOMBlev/WRNLev helpers (lazy PyCHARMM import for unit tests)."""

from __future__ import annotations

from contextlib import contextmanager


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
    pycharmm.lingo.charmm_script(f"bomlev {int(level)}\nwrnlev {int(level)}")
    try:
        yield
    finally:
        settings.set_bomb_level(old_bl)
        settings.set_warn_level(old_wl)
        pycharmm.lingo.charmm_script(f"bomlev {int(old_bl)}\nwrnlev {int(old_wl)}")
