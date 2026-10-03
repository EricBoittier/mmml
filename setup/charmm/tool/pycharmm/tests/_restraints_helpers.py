"""Shared scaffolding for the test_restraints_* split test files.

The original single-file test_restraints.py grew to 20 classes and
1325 lines. It was split by subsystem (noe / resd / atoms / meta)
for navigability; this helper holds the shared `_wipe_psf` helper so
each split file can pull from one place.

Module-private (leading underscore) so pytest doesn't try to collect
it as a test file.
"""

from __future__ import annotations


def wipe_psf() -> None:
    """Clear the PSF if it has any atoms, with warn/bomb levels lowered.

    Used in restraint tests to enforce a clean per-test starting
    state -- many of the restraint setups (NOE, RESD, harm/fix) are
    sensitive to lingering atoms from a previous test that could
    accidentally satisfy a selection.
    """
    from pycharmm import psf, settings

    old_warn = settings.set_warn_level(-5)
    old_bomb = settings.set_bomb_level(-5)
    try:
        if psf.get_natom() > 0:
            psf.delete_atoms()
    finally:
        settings.set_warn_level(old_warn)
        settings.set_bomb_level(old_bomb)
