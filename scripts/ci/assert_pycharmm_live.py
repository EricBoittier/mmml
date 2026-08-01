#!/usr/bin/env python3
"""Fail loudly when ``libcharmm`` is present but not actually usable.

``conftest.can_import_pycharmm`` (and every ``skipif`` built on it) resolves to
``charmm_lib_available()``, which only checks that a ``libcharmm`` *file* exists
under ``CHARMM_LIB_DIR``.  A library that is present but cannot be ``dlopen``'ed
-- wrong ABI, missing ``libgfortran``/``libmpi`` at runtime, a truncated build
restored from cache -- therefore satisfies the guard, and the live-CHARMM CI job
turns into an all-skip run that exits 0.

This script closes that gap: it imports PyCHARMM for real and exits non-zero,
with the underlying loader error, when the import does not work.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths

    home, lib = resolve_charmm_paths()
    print(f"CHARMM_HOME={home or '<unset>'}")
    print(f"CHARMM_LIB_DIR={lib or '<unset>'}")
    if not home or not lib:
        print(
            "::error::assert_pycharmm_live: CHARMM paths did not resolve; "
            "libcharmm was never built or discovered",
            file=sys.stderr,
        )
        return 1

    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import (
            ensure_pycharmm_loaded,
        )

        ensure_pycharmm_loaded()
        import pycharmm  # noqa: F401  (the dlopen actually happens here)
    except BaseException:  # noqa: BLE001 - report anything, including SystemExit
        traceback.print_exc()
        print(
            "::error::assert_pycharmm_live: libcharmm is present but PyCHARMM "
            "could not be loaded; the live-CHARMM suite would have silently "
            "skipped every test",
            file=sys.stderr,
        )
        return 1

    print("assert_pycharmm_live: PyCHARMM loaded successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
