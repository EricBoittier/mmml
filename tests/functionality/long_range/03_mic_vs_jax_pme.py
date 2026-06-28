#!/usr/bin/env python3
"""Step 3: compare MIC (full / truncated) vs jax-pme Ewald."""

from __future__ import annotations

import sys

import numpy as np

from _common import (
    DEFAULT_MM_COULOMB_CUTOFF_A,
    cscl_crystal,
    have_jax_pme_package,
    ion_dimer_system,
    jax_pme_coulomb_energy_forces,
    mic_coulomb_energy_forces,
    print_fail,
    print_header,
    print_pass,
    random_neutral_cluster,
)


def main() -> int:
    print_header("MIC vs jax-pme Ewald")
    if not have_jax_pme_package():
        print_fail("jax-pme not installed")
        return 1

    ok = True

    # Large box dimer: MIC all-pairs ≈ vacuum limit; jax-pme should agree.
    dimer = ion_dimer_system(separation_A=6.0, box_length_A=50.0)
    mic_full = mic_coulomb_energy_forces(dimer, cutoff_A=None)
    ewald = jax_pme_coulomb_energy_forces(dimer, method="ewald", sr_cutoff_A=6.0)
    try:
        np.testing.assert_allclose(mic_full.energy_kcalmol, ewald.energy_kcalmol, rtol=0.05)
        print_pass(
            f"large-box dimer: MIC={mic_full.energy_kcalmol:.4f} "
            f"Ewald={ewald.energy_kcalmol:.4f} kcal/mol"
        )
    except AssertionError as exc:
        print_fail(f"large-box dimer energy: {exc}")
        ok = False

    # Small periodic crystal: MIC all-pairs ≠ full Ewald (missing k-space).
    cscl = cscl_crystal(box_length_A=10.0)
    mic_cscl = mic_coulomb_energy_forces(cscl, cutoff_A=None)
    ewald_cscl = jax_pme_coulomb_energy_forces(cscl, method="ewald", sr_cutoff_A=6.0)
    ratio = mic_cscl.energy_kcalmol / ewald_cscl.energy_kcalmol
    if abs(ratio - 1.0) > 0.01:
        print_pass(
            f"CsCl: MIC ({mic_cscl.energy_kcalmol:.2f}) differs from Ewald "
            f"({ewald_cscl.energy_kcalmol:.2f}) — expected for truncated real-space"
        )
    else:
        print_fail("CsCl: MIC unexpectedly matches Ewald (check test geometry)")

    # Truncated MIC at default MM cutoff underestimates vs Ewald.
    cluster = random_neutral_cluster(n_atoms=8, box_length_A=12.0, seed=7)
    mic_trunc = mic_coulomb_energy_forces(cluster, cutoff_A=DEFAULT_MM_COULOMB_CUTOFF_A)
    ewald_cl = jax_pme_coulomb_energy_forces(cluster, method="ewald", sr_cutoff_A=6.0)
    if abs(mic_trunc.energy_kcalmol) < abs(ewald_cl.energy_kcalmol):
        print_pass(
            f"8-ion cluster: |MIC@{DEFAULT_MM_COULOMB_CUTOFF_A}Å|="
            f"{abs(mic_trunc.energy_kcalmol):.2f} < |Ewald|="
            f"{abs(ewald_cl.energy_kcalmol):.2f} kcal/mol"
        )
    else:
        print_fail("truncated MIC should underestimate |E| vs full Ewald for this cluster")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
