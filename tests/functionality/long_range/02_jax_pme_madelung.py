#!/usr/bin/env python3
"""Step 2: jax-pme Madelung constants (Ewald / PME / P3M) — jax-pme test pattern."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from _common import (
    cscl_crystal,
    have_jax_pme_package,
    jax_pme_coulomb_energy_forces,
    madelung_constant,
    nacl_cubic,
    print_fail,
    print_header,
    print_pass,
)


def main() -> int:
    print_header("jax-pme Madelung constants")
    if not have_jax_pme_package():
        print_fail("jax-pme not installed")
        return 1

    ok = True
    cases = [
        (cscl_crystal(box_length_A=10.0), "ewald", 4e-6),
        (cscl_crystal(box_length_A=10.0), "pme", 9e-4),
        (cscl_crystal(box_length_A=10.0), "p3m", 9e-4),
        (nacl_cubic(box_length_A=5.6), "ewald", 4e-6),
    ]

    for system, method, rtol in cases:
        result = jax_pme_coulomb_energy_forces(system, method=method, sr_cutoff_A=6.0)
        m = madelung_constant(result, system)
        ref = system.madelung_ref
        try:
            np.testing.assert_allclose(m, ref, rtol=rtol)
            print_pass(f"{system.name} {method}: madelung={m:.6f} (ref {ref})")
        except AssertionError as exc:
            print_fail(f"{system.name} {method}: {exc}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
