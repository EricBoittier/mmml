#!/usr/bin/env python
"""Step 04 — the CGenFF lattice energy, and evidence that it is converged.

The lattice energy is the intermolecular energy per molecule: what it costs to
take one acetone out of the crystal and leave the rest behind. Because only
intermolecular terms are summed, everything intramolecular cancels exactly
against an isolated molecule frozen at the same geometry, so no separate gas
calculation is needed.

Two sums have to converge and neither does so for free:

* Dispersion falls off as r^-6, which is slow enough that a bare cutoff leaves
  a visible error. An analytic tail correction absorbs it, and the test is that
  ``LJ + tail`` stops moving as the cutoff grows even though each piece keeps
  moving.
* Electrostatics between dipoles is conditionally convergent -- the answer
  depends on summation order -- so it is done by Ewald rather than truncation.
  The test is that the electrostatic term does not depend on the cutoff at all,
  since the cutoff only sets where work shifts between real and reciprocal
  space.
"""

from __future__ import annotations

import os
import sys

from mmml.analysis.acetone_crystal import ACETONE_CRYSTAL_PHASES, read_acetone_phase
from mmml.analysis.lattice_energy import KCAL_MOL_TO_KJ_MOL, crystal_lattice_energy

PHASE = os.environ.get("ACO_PHASE", "pbca_150k")
CUTOFFS = (8.0, 10.0, 12.0, 14.0, 16.0)

# A converged lattice energy should not move by more than this between the two
# longest cutoffs. Set from the observed behaviour, not from wishful thinking.
CONVERGENCE_TOLERANCE_KCAL = 0.02

phase = ACETONE_CRYSTAL_PHASES[PHASE]
if not phase.usable_for_mm:
    print(f"04: phase {PHASE} has disordered hydrogens; no force field applies.", file=sys.stderr)
    print("    Use ACO_PHASE=pbca_150k (or pbca_110k, pbca_5k).", file=sys.stderr)
    sys.exit(1)

atoms = read_acetone_phase(PHASE)
print(f"=== 04: lattice energy of {PHASE} ===")
print(f"    {phase.label}")
print(f"    Z = {phase.z}, cell {phase.cell_lengths_A[0]:.4f} x {phase.cell_lengths_A[1]:.4f} "
      f"x {phase.cell_lengths_A[2]:.4f} A\n")

print("  cutoff      LJ     tail   LJ+tail    elec    E_latt      shifts    k-vecs")
print("  " + "-" * 72)
results = []
for cutoff in CUTOFFS:
    r = crystal_lattice_energy(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        atoms.cell.array,
        cutoff_A=cutoff,
    )
    results.append(r)
    print(
        f"  {cutoff:5.1f}  {r.e_lj:7.3f} {r.e_lj_tail:7.3f}  {r.e_lj + r.e_lj_tail:8.3f} "
        f"{r.e_coulomb:8.3f} {r.e_lattice:9.3f}   {r.n_lattice_shifts:7d}   {r.n_kvectors:7d}"
    )
print("  (kcal/mol per molecule)\n")

final = results[-1]
print(f"  E_latt = {final.e_lattice:.3f} kcal/mol = "
      f"{final.e_lattice * KCAL_MOL_TO_KJ_MOL:.2f} kJ/mol per molecule")
print(f"  split  : {final.e_lj + final.e_lj_tail:.3f} dispersion/repulsion, "
      f"{final.e_coulomb:.3f} electrostatic "
      f"({100 * final.e_coulomb / final.e_lattice:.0f}% electrostatic)")
print(f"  density: {final.density_g_cm3:.4f} g/cm^3\n")

print("-- convergence --")
lj_drift = abs((results[-1].e_lj + results[-1].e_lj_tail) - (results[0].e_lj + results[0].e_lj_tail))
lj_raw_drift = abs(results[-1].e_lj - results[0].e_lj)
elec_drift = abs(results[-1].e_coulomb - results[0].e_coulomb)
total_drift = abs(results[-1].e_lattice - results[-2].e_lattice)
print(f"  bare LJ moves {lj_raw_drift:.3f} kcal/mol between {CUTOFFS[0]:g} and {CUTOFFS[-1]:g} A,")
print(f"  but LJ + tail moves only {lj_drift:.3f} — the tail correction is doing its job.")
print(f"  Electrostatics moves {elec_drift:.4f}, i.e. not at all: Ewald's answer does")
print(f"  not depend on where the real/reciprocal split is placed.")
print(f"  Between the last two cutoffs the total moves {total_drift:.4f} kcal/mol.\n")

failures = []
if total_drift > CONVERGENCE_TOLERANCE_KCAL:
    failures.append(
        f"lattice energy not converged: {total_drift:.4f} kcal/mol between "
        f"{CUTOFFS[-2]:g} and {CUTOFFS[-1]:g} A"
    )
if elec_drift > 1e-3:
    failures.append(f"Ewald energy depends on the cutoff by {elec_drift:.4f} kcal/mol")
if final.e_lattice > 0:
    failures.append(f"lattice energy is positive ({final.e_lattice:.3f}); the crystal is unbound")

if failures:
    for f in failures:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("04: FAILED", file=sys.stderr)
    sys.exit(1)
print("04: OK")
