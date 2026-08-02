#!/usr/bin/env python
"""Step 05 — sublimation enthalpy, across the three Pbca temperatures.

    dH_sub(T) = -E_latt - 2RT

The ``-2RT`` is the difference between what a molecule carries in the gas
(3/2 RT translation + 3/2 RT rotation + RT for pV = 4RT) and what replaces it
in the crystal (six rigid-body lattice modes, each worth RT classically = 6RT).
It assumes the molecule is rigid and that intramolecular vibration is the same
in both phases, and it is classical throughout -- no zero-point term. At 5 K
that last assumption is badly violated in principle, though the correction
enters as a difference of zero-point energies between phases and stays small.

Two comparisons are worth making, and they answer different questions:

* Against experiment, which asks whether CGenFF describes solid acetone.
* Across temperature, which asks whether the model reproduces the *trend* Allan
  et al. report -- contacts strengthening on cooling. That comparison is
  internal to the force field and so is not contaminated by its absolute error.
"""

from __future__ import annotations

import os
import sys

from mmml.analysis.acetone_crystal import (
    ACETONE_CRYSTAL_PHASES,
    ACETONE_SUBLIMATION_REFERENCE,
    read_acetone_phase,
)
from mmml.analysis.lattice_energy import KCAL_MOL_TO_KJ_MOL, crystal_lattice_energy

CUTOFF = float(os.environ.get("ACO_CUTOFF", "12.0"))
SCALES_FILE = os.environ.get("ACO_SCALES", "").strip()

# Deliberately wide. The experimental value is itself assembled from a cycle at
# ~180 K while the structures sit at 5-150 K, and CGenFF was never fit to a
# crystal. This bound catches a broken calculation, not a mediocre force field.
PLAUSIBLE_KJ_MOL = (25.0, 60.0)

sigma_scale = epsilon_scale = None
if SCALES_FILE:
    from mmml.models.mm_lj_scales import load_mm_lj_scales_sidecar

    payload = load_mm_lj_scales_sidecar(SCALES_FILE)
    if payload is None:
        print(f"05: no learnable LJ scales found in {SCALES_FILE}", file=sys.stderr)
        sys.exit(1)
    sigma_scale = payload.get("sigma_scale")
    epsilon_scale = payload.get("epsilon_scale")
    print(f"=== 05: sublimation enthalpy (LJ scales from {SCALES_FILE}) ===\n")
else:
    print("=== 05: sublimation enthalpy (stock CGenFF) ===\n")

reference = ACETONE_SUBLIMATION_REFERENCE
print("Experimental reference, assembled through a thermodynamic cycle because")
print("no direct sublimation measurement for acetone is tabulated:")
print(
    f"  dH_vap = {reference.dvap_h_kj_mol:.1f} kJ/mol at {reference.dvap_h_temperature_K:.0f} K"
    f"   [{reference.dvap_h_source}]"
)
print(
    f"  dH_fus = {reference.dfus_h_kj_mol:.2f} kJ/mol at {reference.dfus_h_temperature_K:.1f} K"
    f"   [{reference.dfus_h_source}]"
)
print(
    f"  dH_sub ~ {reference.dsub_h_kj_mol:.1f} kJ/mol "
    f"({reference.dsub_h_kcal_mol:.2f} kcal/mol) near the melting point\n"
)

print("  phase        T/K    E_latt    -2RT    dH_sub       dH_sub    vs experiment")
print("                     kcal/mol  kcal/mol  kcal/mol     kJ/mol")
print("  " + "-" * 76)

rows = []
for key in ("pbca_150k", "pbca_110k", "pbca_5k"):
    phase = ACETONE_CRYSTAL_PHASES[key]
    atoms = read_acetone_phase(key)
    result = crystal_lattice_energy(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        atoms.cell.array,
        cutoff_A=CUTOFF,
        sigma_scale=sigma_scale,
        epsilon_scale=epsilon_scale,
    )
    dh = result.sublimation_enthalpy(phase.temperature_K)
    dh_kj = dh * KCAL_MOL_TO_KJ_MOL
    error = 100.0 * (dh_kj - reference.dsub_h_kj_mol) / reference.dsub_h_kj_mol
    rows.append((phase, result, dh_kj))
    print(
        f"  {key:11s} {phase.temperature_K:5.0f}  {result.e_lattice:8.3f}  "
        f"{dh - (-result.e_lattice):7.3f}  {dh:8.3f}  {dh_kj:11.2f}    {error:+6.1f}%"
    )
print()

print("-- reading the numbers --")
warmest, coldest = rows[0][2], rows[-1][2]
print(f"  dH_sub rises from {warmest:.1f} kJ/mol at 150 K to {coldest:.1f} kJ/mol at 5 K,")
print(f"  a {100 * (coldest - warmest) / warmest:.0f}% increase. That is the force field agreeing with the")
print("  paper's central claim: the contacts really are stronger in the colder")
print("  cells, and the effect is large enough to see in a static energy.")
print()
print("  Each row is evaluated at its own experimental geometry, so this is not")
print("  a thermal-expansion model -- it is the energy of three measured")
print("  structures. The trend is inherited from the diffraction data.")
print()
mean_error = sum(row[2] for row in rows) / len(rows) - reference.dsub_h_kj_mol
print(f"  Against experiment CGenFF overbinds by roughly {mean_error:.0f} kJ/mol. That is a")
print("  familiar result for a force field parameterised against liquid-phase")
print("  densities and heats of vaporisation: nothing in its fit ever saw a")
print("  crystal, and the temperature mismatch works the same way, since the")
print("  reference sits near 180 K and these structures are colder.")
print()

if SCALES_FILE:
    print("  These numbers used learned LJ scales. Compare against a stock run")
    print("  (unset ACO_SCALES) to see whether training moved the crystal in the")
    print("  right direction -- sublimation enthalpy is an observable the fit")
    print("  never saw, so it is a genuine test rather than a re-read of training")
    print("  loss.\n")

failures = []
for phase, result, dh_kj in rows:
    if not (PLAUSIBLE_KJ_MOL[0] <= dh_kj <= PLAUSIBLE_KJ_MOL[1]):
        failures.append(
            f"{phase.key}: dH_sub {dh_kj:.1f} kJ/mol outside the plausible range "
            f"{PLAUSIBLE_KJ_MOL}"
        )
if rows[-1][2] <= rows[0][2]:
    failures.append(
        "dH_sub does not increase on cooling, contradicting the measured contacts"
    )

if failures:
    for f in failures:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("05: FAILED", file=sys.stderr)
    sys.exit(1)
print("05: OK")
