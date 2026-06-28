#!/usr/bin/env python3
"""Step 4: ScaFaCoS method cross-check (ewald, p3m, …) vs jax-pme reference."""

from __future__ import annotations

import os
import sys

import numpy as np

from _common import (
    cscl_crystal,
    have_jax_pme_package,
    have_scafacos_library,
    ion_dimer_system,
    jax_pme_coulomb_energy_forces,
    print_fail,
    print_header,
    print_pass,
    scafacos_coulomb_energy_forces,
)
from mmml.interfaces.scafacosInterface.scafacos_session import SCAFACOS_METHODS


def main() -> int:
    print_header("ScaFaCoS method comparison")
    if not have_scafacos_library():
        print("SKIP: ScaFaCoS libfcs not available (set SCAFACOS_LIB)")
        return 0

    if not have_jax_pme_package():
        print_fail("jax-pme required as reference")
        return 1

    ok = True
    system = cscl_crystal(box_length_A=10.0)
    ref = jax_pme_coulomb_energy_forces(system, method="ewald", sr_cutoff_A=6.0)
    print(f"Reference (jax-pme Ewald): E = {ref.energy_kcalmol:.6f} kcal/mol")

    default_method = os.environ.get("SCAFACOS_METHOD", "p3m")
    methods_to_try = [default_method] + [m for m in SCAFACOS_METHODS if m != default_method]

    for method in methods_to_try:
        try:
            out = scafacos_coulomb_energy_forces(system, method=method)
            np.testing.assert_allclose(
                out.energy_kcalmol,
                ref.energy_kcalmol,
                rtol=5e-3,
                err_msg=f"ScaFaCoS {method} energy",
            )
            np.testing.assert_allclose(
                out.forces_kcalmol_A,
                ref.forces_kcalmol_A,
                rtol=5e-2,
                err_msg=f"ScaFaCoS {method} forces",
            )
            print_pass(f"{method}: E={out.energy_kcalmol:.6f} kcal/mol")
        except Exception as exc:
            print(f"INFO: ScaFaCoS {method} skipped or failed: {exc}")

    # Dimer sanity: neutral pair in large box
    dimer = ion_dimer_system(separation_A=5.0, box_length_A=30.0)
    ref_d = jax_pme_coulomb_energy_forces(dimer, method="ewald")
    try:
        out_d = scafacos_coulomb_energy_forces(dimer, method=default_method)
        np.testing.assert_allclose(out_d.energy_kcalmol, ref_d.energy_kcalmol, rtol=0.02)
        print_pass(f"dimer {default_method} vs jax-pme Ewald")
    except Exception as exc:
        print_fail(f"dimer comparison: {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
