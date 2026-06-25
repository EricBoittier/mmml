#!/usr/bin/env python3
"""Step 0: check neighbor-list validation environment (jax-md, optional vesin)."""

from __future__ import annotations

import sys

from _common import print_fail, print_header, print_pass


def main() -> int:
    print_header("Neighbor-list validation environment")
    ok = True

    try:
        import numpy  # noqa: F401
        print_pass("numpy")
    except ImportError as exc:
        print_fail(f"numpy: {exc}")
        ok = False

    try:
        import jax  # noqa: F401
        print_pass("jax")
    except ImportError as exc:
        print_fail(f"jax: {exc}")
        ok = False

    try:
        from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import have_jax_md

        if have_jax_md():
            print_pass("jax-md neighbor lists")
        else:
            print_fail("jax-md import succeeded but have_jax_md() is False")
            ok = False
    except ImportError as exc:
        print_fail(f"jax-md: {exc}")
        ok = False

    try:
        from mmml.interfaces.pycharmmInterface.nl_reference import have_vesin

        if have_vesin():
            print_pass("vesin (reference oracle)")
        else:
            print("INFO: vesin not installed — scripts use brute-force reference")
            print("      Install: uv sync --extra nl-validation")
    except ImportError as exc:
        print_fail(f"nl_reference: {exc}")
        ok = False

    try:
        from mmml.interfaces.pycharmmInterface.nl_backend import (
            CellListBackend,
            VesinBackend,
        )
        from mmml.interfaces.pycharmmInterface.nl_reference import have_vesin as _have_vesin

        print_pass("nl_backend module")
        _ = CellListBackend()
        if _have_vesin():
            _ = VesinBackend()
    except ImportError as exc:
        print_fail(f"nl_backend: {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
