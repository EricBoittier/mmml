#!/usr/bin/env python3
"""Step 0: probe long-range Coulomb backends (MIC, jax-pme, ScaFaCoS)."""

from __future__ import annotations

import sys

from _common import (
    describe_environment,
    have_jax_pme_package,
    have_scafacos_library,
    print_fail,
    print_header,
    print_pass,
)


def main() -> int:
    print_header("Long-range Coulomb validation environment")
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
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

        _ = mic_displacement
        print_pass("MIC utilities (pbc_utils_jax)")
    except ImportError as exc:
        print_fail(f"MIC utilities: {exc}")
        ok = False

    if have_jax_pme_package():
        print_pass("jax-pme (jaxpme)")
    else:
        print_fail("jax-pme not installed — install mmml deps or: pip install jax-pme")
        ok = False

    if have_scafacos_library():
        print_pass("ScaFaCoS libfcs")
    else:
        print("INFO: ScaFaCoS not found — ScaFaCoS comparison scripts will skip")
        print("      Set SCAFACOS_LIB=/path/to/libfcs.so after building ScaFaCoS")

    print("\n" + describe_environment())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
