#!/usr/bin/env python
"""Step 03 — testing the paper's conclusion: H...Cl or Cl...Cl?

Podsiadło et al. close with a claim about energy, not geometry:

    "the crystal cohesion forces are dominated by H...Cl interactions rather
     than by Cl...Cl attractions"

They reached it indirectly, from how the contacts compress and from the crystal
habit, because a diffraction experiment measures positions and not energies.
A force field measures energies directly, so the claim can simply be checked.

The decomposition is over *molecule* pairs, each labelled by its shortest
contact. Molecule pairs rather than atom pairs because each CH2Cl2 is neutral,
so a molecule-pair energy is a well-defined interaction energy; an atom-pair
split of the same lattice is dominated by cancelling monopole terms of order
100 kcal/mol that mean nothing individually.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*crystal system.*")

from mmml.analysis.dcm_crystal import (  # noqa: E402
    DCM_CRYSTAL_PHASES,
    halogen_contacts,
    read_dcm_phase,
)
from mmml.analysis.lattice_energy import (  # noqa: E402
    decompose_lattice_energy_by_element_pair,
)

CUTOFF_A = float(os.environ.get("DCM_CUTOFF", "12.0"))

FAIL: list[str] = []

print("=== 03: what holds the crystal together ===")
print()
print("Claim under test (Podsiadlo, Dziubek & Katrusiak 2005):")
print('  "the crystal cohesion forces are dominated by H...Cl interactions')
print('   rather than by Cl...Cl attractions"')
print()

for key, phase in DCM_CRYSTAL_PHASES.items():
    atoms = read_dcm_phase(key, rebuild_hydrogens=True)
    dec = decompose_lattice_energy_by_element_pair(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        atoms.cell.array,
        cutoff_A=CUTOFF_A,
    )
    print(f"{key}  ({phase.pressure_GPa:g} GPa)")
    print("   contact     dispersion  electrostatic     total     share")
    print("   ---------------------------------------------------------")
    for contact, lj, coul, total, _ in dec.ranked():
        label = "...".join(contact)
        share = 100.0 * total / dec.e_total
        print(
            f"   {label:10s} {lj:10.3f} {coul:14.3f} {total:9.3f}   {share:5.1f}%"
        )
    print(f"   {'total':10s} {dec.e_lj:10.3f} {dec.e_coulomb_direct:14.3f} {dec.e_total:9.3f}")
    print("   (kcal/mol per molecule)")
    print(
        f"   direct Coulomb sum differs from Ewald by "
        f"{dec.coulomb_truncation_error:+.4f} kcal/mol"
    )

    ranked = dec.ranked()
    dominant = ranked[0]
    hcl = next((row for row in ranked if set(row[0]) == {"H", "Cl"}), None)
    clcl = next((row for row in ranked if set(row[0]) == {"Cl"}), None)
    if hcl is None or clcl is None:
        FAIL.append(f"{key}: expected both H...Cl and Cl...Cl buckets")
        continue
    if set(dominant[0]) != {"H", "Cl"}:
        FAIL.append(
            f"{key}: the paper says H...Cl dominates, but the most attractive "
            f"bucket is {'...'.join(dominant[0])}"
        )
    if abs(clcl[3]) >= abs(hcl[3]):
        FAIL.append(
            f"{key}: Cl...Cl ({clcl[3]:.3f}) is not weaker than H...Cl ({hcl[3]:.3f})"
        )
    if abs(dec.coulomb_truncation_error) > 0.05:
        FAIL.append(
            f"{key}: direct Coulomb sum is {dec.coulomb_truncation_error:+.3f} "
            f"kcal/mol from Ewald, too far for the split to be trusted"
        )
    print(
        f"   -> H...Cl carries {100.0 * hcl[3] / dec.e_total:.0f}% of the binding, "
        f"Cl...Cl {100.0 * clcl[3] / dec.e_total:.0f}%"
    )
    print()

# --- the geometric picture disagrees, and that is the interesting part -------
atoms = read_dcm_phase("pbcn_133gpa", rebuild_hydrogens=True)
closest_halogen = halogen_contacts(atoms)[0]
dec = decompose_lattice_energy_by_element_pair(
    atoms.get_positions(),
    atoms.get_atomic_numbers(),
    atoms.cell.array,
    cutoff_A=CUTOFF_A,
)
clcl = next(row for row in dec.ranked() if set(row[0]) == {"Cl"})

print("-- reading the numbers --")
print("  CGenFF agrees with the paper, and the reason is visible in the split.")
print(
    f"  The shortest Cl...Cl contact is {closest_halogen.distance_A:.3f} A, inside the"
)
print(
    f"  3.50 A van der Waals sum, and its geometry is Type {closest_halogen.motif}:"
)
print("  by the standard geometric criteria this is a halogen bond. But its")
print(
    f"  electrostatic contribution is {clcl[2]:+.3f} kcal/mol -- repulsive -- and all"
)
print(f"  {clcl[1]:.3f} kcal/mol of its binding is dispersion.")
print()
print("  CGenFF has no sigma-hole: chlorine carries a single point charge, so")
print("  the model cannot produce an attractive halogen bond even in principle.")
print("  What it can say is that you do not need one to hold this crystal")
print("  together, because the contact is bound by dispersion regardless. A")
print("  later plane-wave DFT study, which does describe the sigma hole,")
print("  reached the same conclusion: Kurzydlowski, Chumak & Rogoza, Crystals")
print("  10, 920 (2020), find halogen bonds play 'only a minor role' in CH2Cl2")
print("  while hydrogen bonds and dipole-dipole terms dominate.")

print()
if FAIL:
    for f in FAIL:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("\n03: FAILED", file=sys.stderr)
    sys.exit(1)
print("03: OK — H...Cl dominates cohesion in both structures, as published")
