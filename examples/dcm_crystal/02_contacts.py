#!/usr/bin/env python
"""Step 02 — the close contacts, and the hydrogen problem underneath them.

The paper is about two families of contact: Cl...Cl halogen contacts and
C-H...Cl hydrogen bonds. This step measures both in each deposited structure and
classifies the Cl...Cl ones by the Desiraju-Parthasarathy geometry.

It also demonstrates why the deposited hydrogens cannot be used as they stand.
The two refinements put C-H at 1.01(10) and 1.13(12) A and disagree on the
hydrogen *direction* by a comparable amount; both are ordinary X-ray behaviour
and both are larger than the compression between the two structures. Left alone
they make the shortest H...Cl contact appear to *lengthen* under pressure.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*crystal system.*")

from mmml.analysis.dcm_crystal import (  # noqa: E402
    DCM_CRYSTAL_PHASES,
    h_cl_contacts,
    halogen_contacts,
    read_dcm_phase,
    rebuild_methylene_hydrogens,
)

# Sum of the van der Waals radii, Bondi 1964: a contact shorter than this is
# conventionally "close".
VDW_CL_CL_A = 3.50
VDW_H_CL_A = 2.95

FAIL: list[str] = []

print("=== 02: intermolecular contacts ===")
print()

shortest_raw: dict[str, float] = {}
shortest_rebuilt: dict[str, float] = {}

for key, phase in DCM_CRYSTAL_PHASES.items():
    deposited = read_dcm_phase(key)
    rebuilt = rebuild_methylene_hydrogens(deposited)

    print(f"{key}  ({phase.label})")
    print("  Cl...Cl contacts")
    halogens = halogen_contacts(deposited, max_distance_A=4.2)
    if not halogens:
        FAIL.append(f"{key}: found no Cl...Cl contacts at all")
    for contact in halogens[:4]:
        flag = " <- inside vdW" if contact.distance_A < VDW_CL_CL_A else ""
        print(
            f"     {contact.distance_A:6.3f} A   largest C-Cl...Cl angle "
            f"{contact.angle_deg:6.1f} deg   Type {contact.motif}{flag}"
        )

    print("  H...Cl contacts (deposited hydrogens vs rebuilt)")
    raw = h_cl_contacts(deposited, rebuild_hydrogens=False)
    fixed = h_cl_contacts(deposited, rebuild_hydrogens=True)
    if not raw or not fixed:
        FAIL.append(f"{key}: found no H...Cl contacts at all")
        continue
    shortest_raw[key] = raw[0].distance_A
    shortest_rebuilt[key] = fixed[0].distance_A
    for i in range(min(3, len(raw), len(fixed))):
        flag = " <- inside vdW" if fixed[i].distance_A < VDW_H_CL_A else ""
        print(
            f"     deposited {raw[i].distance_A:6.3f} A     "
            f"rebuilt {fixed[i].distance_A:6.3f} A{flag}"
        )
    print()

# --- what the hydrogens do to the comparison ---------------------------------
print("-- why the hydrogens had to be rebuilt --")
keys = list(DCM_CRYSTAL_PHASES)
lo, hi = keys[0], keys[1]
p_lo = DCM_CRYSTAL_PHASES[lo].pressure_GPa
p_hi = DCM_CRYSTAL_PHASES[hi].pressure_GPa
d_raw = shortest_raw[hi] - shortest_raw[lo]
d_fixed = shortest_rebuilt[hi] - shortest_rebuilt[lo]
print(f"  Going from {p_lo:g} to {p_hi:g} GPa the cell volume falls 2.6%, so every")
print("  contact should shorten. The shortest H...Cl contact does this:")
print(
    f"      deposited hydrogens   {shortest_raw[lo]:.3f} -> {shortest_raw[hi]:.3f} A "
    f"({d_raw:+.3f})"
)
print(
    f"      rebuilt hydrogens     {shortest_rebuilt[lo]:.3f} -> "
    f"{shortest_rebuilt[hi]:.3f} A ({d_fixed:+.3f})"
)
print()
print("  The deposited hydrogens get the sign wrong. They are not wrong in any")
print("  interesting sense -- X-rays scatter from electrons, a hydrogen has one")
print("  bonding electron, and the refined C-H distances came out at 1.01(10)")
print("  and 1.13(12) A. The uncertainty is simply larger than the effect.")
print()
print("  The fix is not a fudge: CH2Cl2 is C2v and its carbon and chlorines are")
print("  located to a few thousandths of an Angstrom, so the hydrogens follow")
print("  from the heavy-atom frame plus two spectroscopic constants that are")
print("  known far better than any diffraction experiment can place a hydrogen.")

if d_raw < 0.0:
    FAIL.append(
        "deposited hydrogens no longer reverse the contact trend -- the point "
        "this step is making has gone stale and the text needs rewriting"
    )
if d_fixed > 0.0:
    FAIL.append(
        f"rebuilt hydrogens still lengthen the shortest H...Cl under "
        f"compression ({d_fixed:+.3f} A), which is unphysical"
    )

print()
if FAIL:
    for f in FAIL:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("\n02: FAILED", file=sys.stderr)
    sys.exit(1)
print("02: OK — contacts measured, hydrogens rebuilt")
