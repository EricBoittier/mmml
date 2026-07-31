#!/usr/bin/env python
"""Step 02 — reproduce the intermolecular contacts the paper is about.

Allan et al. characterise acetone through dipolar carbonyl-carbonyl contacts,
which fall into the three archetypes catalogued by Allen et al. (*Acta
Crystallogr.* B54, 320): antiparallel (Type II), perpendicular (Type I) and
sheared-parallel (Type III). Their explanation for the long-standing heat
capacity anomaly is that these, together with C-H...O contacts, shorten
measurably on cooling.

Recomputing those distances from the expanded cell is the sharpest check that
the structure was assembled correctly. A cell can have the right lattice
parameters and the right number of molecules while still being wrong -- a
mis-applied symmetry operator, or molecules broken across a face -- and the
contact distances catch that immediately, because they are what the authors
measured.
"""

from __future__ import annotations

import sys

from mmml.analysis.acetone_crystal import (
    ACETONE_CRYSTAL_PHASES,
    carbonyl_contacts,
    ch_o_contacts,
    read_acetone_phase,
)

# The paper quotes these to three decimals with uncertainties in the last digit
# or two; agreement to 0.01 A means we are reading the same structure.
MATCH_TOLERANCE_A = 0.01

print("=== 02: intermolecular contacts vs the published values ===\n")

failures: list[str] = []
matched_total = 0

for key, phase in ACETONE_CRYSTAL_PHASES.items():
    atoms = read_acetone_phase(key)
    print(f"{key}  ({phase.label})")

    # Carbonyl geometry needs only C and O, so the disordered-methyl phase is
    # still fair game here even though it cannot carry a force field.
    contacts = carbonyl_contacts(atoms, max_distance_A=3.8)
    print("  carbonyl C...O contacts")
    for contact in contacts:
        print(
            f"    {contact.distance_A:6.3f} A   C=O/C=O angle {contact.angle_deg:6.1f} deg"
            f"   Type {contact.motif}"
        )
    if not contacts:
        print("    (none within 3.8 A)")

    if phase.usable_for_mm:
        hydrogen_bonds = ch_o_contacts(atoms, max_distance_A=2.9)
        print("  C-H...O contacts (H...O separation)")
        for contact in hydrogen_bonds[:4]:
            print(f"    {contact.distance_A:6.3f} A")
    else:
        hydrogen_bonds = []
        print("  C-H...O contacts: skipped, hydrogens are disordered")

    # Match each published distance against the nearest computed one.
    computed = [c.distance_A for c in contacts] + [c.distance_A for c in hydrogen_bonds]
    print("  published:")
    for name, published in sorted(phase.published_contacts.items(), key=lambda kv: kv[1]):
        nearest = min(computed, key=lambda d: abs(d - published)) if computed else float("nan")
        delta = abs(nearest - published)
        verdict = "match" if delta <= MATCH_TOLERANCE_A else f"MISSED by {delta:.3f} A"
        print(f"    {name:32s} {published:6.3f} A   found {nearest:6.3f} A   {verdict}")
        if delta <= MATCH_TOLERANCE_A:
            matched_total += 1
        else:
            failures.append(f"{key}/{name}: published {published:.3f}, nearest {nearest:.3f}")
    print()

print("-- the paper's thermal argument, in these numbers --")
for key in ("pbca_150k", "pbca_110k", "pbca_5k"):
    phase = ACETONE_CRYSTAL_PHASES[key]
    contacts = ch_o_contacts(read_acetone_phase(key), max_distance_A=2.9)
    shortest = ", ".join(f"{c.distance_A:.3f}" for c in contacts[:2])
    print(f"  {phase.temperature_K:5.0f} K   two shortest H...O: {shortest} A")
print("  The contacts tighten monotonically on cooling, which is the structural")
print("  change Allan et al. propose as the origin of the broad heat-capacity")
print("  anomaly near 127 K that has been unexplained since Kelley measured it")
print("  in 1929. Note the 5 K row is neutron-derived, so its H positions are")
print("  physically longer C-H bonds than the X-ray rows; part of that step is")
print("  method, not temperature.\n")

if failures:
    for f in failures:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("02: FAILED", file=sys.stderr)
    sys.exit(1)
print(f"02: OK — {matched_total} published distances reproduced to {MATCH_TOLERANCE_A} A")
