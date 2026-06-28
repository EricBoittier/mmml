#!/usr/bin/env python3
"""Step 1: MIC pair Coulomb vs analytic 1/r for a vacuum-like large box."""

from __future__ import annotations

import sys

import numpy as np

from _common import (
    ion_dimer_system,
    mic_coulomb_energy_forces,
    print_fail,
    print_header,
    print_pass,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import CHARMM_COULOMB_KCAL


def main() -> int:
    print_header("MIC Coulomb vs analytic ion dimer")
    ok = True

    system = ion_dimer_system(separation_A=8.0, box_length_A=60.0)
    result = mic_coulomb_energy_forces(system, cutoff_A=None)
    r = 8.0
    e_analytic = -CHARMM_COULOMB_KCAL / r
    f_analytic = CHARMM_COULOMB_KCAL / r**2

    try:
        np.testing.assert_allclose(result.energy_kcalmol, e_analytic, rtol=1e-10)
        print_pass(f"energy {result.energy_kcalmol:.6f} ≈ {e_analytic:.6f} kcal/mol")
    except AssertionError as exc:
        print_fail(str(exc))
        ok = False

    try:
        np.testing.assert_allclose(result.forces_kcalmol_A[0, 0], f_analytic, rtol=1e-8)
        np.testing.assert_allclose(result.forces_kcalmol_A[1, 0], -f_analytic, rtol=1e-8)
        print_pass("forces match ±q/r² along x")
    except AssertionError as exc:
        print_fail(str(exc))
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
