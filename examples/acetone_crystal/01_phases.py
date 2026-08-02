#!/usr/bin/env python
"""Step 01 — the five phases of Allan et al., and what the deposited CIFs give.

The paper reports acetone in two distinct phases across five structures: a
C-centred Cmcm phase (at 15 kbar, and metastable at low temperature) and the
stable primitive Pbca phase, measured at 150, 110 and 5 K. This step reads each
deposited cell and checks it against the numbers quoted in the paper's
footnotes, so a later lattice energy is not computed on a structure nobody
looked at.

The check that matters most is Z. ASE applies the deposited symmetry operators,
so a Pbca entry must come back as 16 whole molecules rather than the
two-molecule asymmetric unit. If that expansion silently failed, every number
downstream would be wrong by a factor of eight.
"""

from __future__ import annotations

import sys

import numpy as np

from mmml.analysis.acetone_crystal import ACETONE_CRYSTAL_PHASES, read_acetone_phase
from mmml.analysis.lattice_energy import unwrap_molecules

CELL_TOLERANCE_A = 0.01
VOLUME_TOLERANCE_FRAC = 0.005

print("=== 01: the published phases ===\n")
print("Allan, Clark, Ibberson, Parsons, Pulham & Sawyer,")
print("Chem. Commun. 1999, 751 (doi:10.1039/a900558g), CCDC 182/1197.\n")

failures: list[str] = []

for key, phase in ACETONE_CRYSTAL_PHASES.items():
    atoms = read_acetone_phase(key)
    lengths = np.asarray(atoms.cell.lengths())
    volume = atoms.get_volume()
    density = sum(atoms.get_masses()) / volume / 0.6022140857

    mol_id, _ = unwrap_molecules(
        atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array
    )
    n_molecules = int(mol_id.max()) + 1
    sizes = np.bincount(mol_id)

    print(f"{key}  (COD {phase.cod_id})")
    print(f"  {phase.label}")
    print(f"  conditions   T = {phase.temperature_K:g} K, p = {phase.pressure_kbar:g} kbar")
    print(
        f"  cell         a={lengths[0]:.4f} b={lengths[1]:.4f} c={lengths[2]:.4f} A"
        f"   (paper: {phase.cell_lengths_A[0]:.4f} {phase.cell_lengths_A[1]:.4f} "
        f"{phase.cell_lengths_A[2]:.4f})"
    )
    print(f"  volume       {volume:.2f} A^3   (paper: {phase.cell_volume_A3:.2f})")
    print(f"  contents     {len(atoms)} atoms, {n_molecules} molecules of {sizes[0]} atoms")
    print(f"  density      {density:.4f} g/cm^3 (protiated masses)")
    if phase.deuterated:
        print("               (refined on acetone-d6; masses reset to H for comparability)")
    if not phase.usable_for_mm:
        print("  NOTE         disordered hydrogens — packing only, not force-field ready")
    print(f"  {phase.note}")
    print()

    deviation = np.abs(lengths - np.asarray(phase.cell_lengths_A))
    if deviation.max() > CELL_TOLERANCE_A:
        failures.append(f"{key}: cell differs from the paper by {deviation.max():.4f} A")
    if abs(volume - phase.cell_volume_A3) / phase.cell_volume_A3 > VOLUME_TOLERANCE_FRAC:
        failures.append(f"{key}: volume {volume:.2f} vs published {phase.cell_volume_A3:.2f}")
    if n_molecules != phase.z:
        failures.append(
            f"{key}: symmetry expansion gave Z={n_molecules}, paper says Z={phase.z}"
        )
    expected_atoms = 10 if phase.usable_for_mm else 16  # disordered methyls carry 12 H
    if not np.all(sizes == expected_atoms):
        failures.append(f"{key}: molecules are not all {expected_atoms} atoms: {set(sizes)}")

print("-- what the cells say about the paper's argument --")
print("  Cooling the Pbca phase from 150 K to 5 K is strongly anisotropic:")
for key in ("pbca_150k", "pbca_110k", "pbca_5k"):
    p = ACETONE_CRYSTAL_PHASES[key]
    a, b, c = p.cell_lengths_A
    print(f"    {p.temperature_K:5.0f} K   a={a:7.4f}  b={b:7.4f}  c={c:8.4f}  V={p.cell_volume_A3:7.1f}")
print("  a expands by 3% while b contracts by 6% and c by 4%. A cubic box cannot")
print("  represent that, which is why this ladder carries the full cell throughout.\n")

if failures:
    for f in failures:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("01: FAILED", file=sys.stderr)
    sys.exit(1)
print("01: OK — all five cells match the published values")
