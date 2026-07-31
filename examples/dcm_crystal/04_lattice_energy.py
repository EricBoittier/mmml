#!/usr/bin/env python
"""Step 04 — the CGenFF lattice energy, and whether it is converged.

The intermolecular energy per molecule of the deposited cell, summed explicitly
over lattice images with an Ewald electrostatic term and an analytic dispersion
tail. The point of the cutoff sweep is that you should not have to take 12 A on
faith: the table shows what each term does as the cutoff grows.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*crystal system.*")

from mmml.analysis.dcm_crystal import DCM_CRYSTAL_PHASES, read_dcm_phase  # noqa: E402
from mmml.analysis.lattice_energy import (  # noqa: E402
    KCAL_MOL_TO_KJ_MOL,
    crystal_lattice_energy,
)

PHASE = os.environ.get("DCM_PHASE", "pbcn_133gpa")
CUTOFF_A = float(os.environ.get("DCM_CUTOFF", "12.0"))
SWEEP_A = (8.0, 10.0, 12.0, 14.0, 16.0)

FAIL: list[str] = []

phase = DCM_CRYSTAL_PHASES[PHASE]
atoms = read_dcm_phase(PHASE, rebuild_hydrogens=True)

print(f"=== 04: lattice energy of {PHASE} ===")
print(f"    {phase.label}")
a, b, c = atoms.cell.lengths()
print(f"    Z = {phase.z}, cell {a:.4f} x {b:.4f} x {c:.4f} A, p = {phase.pressure_GPa:g} GPa")
print()
print("  cutoff      LJ     tail   LJ+tail    elec    E_latt      shifts    k-vecs")
print("  ------------------------------------------------------------------------")

results = {}
for cutoff in SWEEP_A:
    r = crystal_lattice_energy(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        atoms.cell.array,
        cutoff_A=cutoff,
    )
    results[cutoff] = r
    print(
        f"  {cutoff:5.1f} {r.e_lj:8.3f} {r.e_lj_tail:8.3f} {r.e_lj + r.e_lj_tail:9.3f} "
        f"{r.e_coulomb:8.3f} {r.e_lattice:9.3f}   {r.n_lattice_shifts:7d} {r.n_kvectors:8d}"
    )
print("  (kcal/mol per molecule)")
print()

chosen = results[CUTOFF_A] if CUTOFF_A in results else crystal_lattice_energy(
    atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=CUTOFF_A
)
e_latt = chosen.e_lattice
dispersion = chosen.e_lj + chosen.e_lj_tail
print(f"  E_latt = {e_latt:.3f} kcal/mol = {e_latt * KCAL_MOL_TO_KJ_MOL:.2f} kJ/mol per molecule")
print(
    f"  split  : {dispersion:.3f} dispersion/repulsion, {chosen.e_coulomb:.3f} "
    f"electrostatic ({100.0 * chosen.e_coulomb / e_latt:.0f}% electrostatic)"
)
print(f"  density: {chosen.density_g_cm3:.4f} g/cm^3")
print()

# --- convergence -------------------------------------------------------------
lo, hi = SWEEP_A[0], SWEEP_A[-1]
bare_lj_drift = abs(results[hi].e_lj - results[lo].e_lj)
tailed_drift = abs(
    (results[hi].e_lj + results[hi].e_lj_tail) - (results[lo].e_lj + results[lo].e_lj_tail)
)
elec_drift = abs(results[hi].e_coulomb - results[lo].e_coulomb)
last_step = abs(results[SWEEP_A[-1]].e_lattice - results[SWEEP_A[-2]].e_lattice)

print("-- convergence --")
print(f"  bare LJ moves {bare_lj_drift:.3f} kcal/mol between {lo:g} and {hi:g} A,")
print(f"  but LJ + tail moves only {tailed_drift:.3f} — the tail correction is working.")
print(f"  Electrostatics moves {elec_drift:.4f}: Ewald does not care where the")
print("  real/reciprocal split is placed, which is the whole point of using it.")
print(f"  Between the last two cutoffs the total moves {last_step:.4f} kcal/mol.")

if tailed_drift > 0.05:
    FAIL.append(f"LJ + tail drifts {tailed_drift:.3f} kcal/mol over the sweep — not converged")
if elec_drift > 1e-3:
    FAIL.append(f"Ewald energy drifts {elec_drift:.4f} kcal/mol with cutoff — that is a bug")
if last_step > 0.01:
    FAIL.append(f"total still moving {last_step:.4f} kcal/mol at the longest cutoffs")
if e_latt >= 0.0:
    FAIL.append(f"lattice energy {e_latt:.3f} kcal/mol is not bound")

print()
if FAIL:
    for f in FAIL:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("\n04: FAILED", file=sys.stderr)
    sys.exit(1)
print("04: OK")
