#!/usr/bin/env python
"""Step 01 — do the built cells match what Podsiadło et al. published?

Reads both deposited structures, expands the deposited symmetry, and checks the
cell, the volume, the molecule count and the composition against the paper.

Also states plainly what is *not* available: the ambient-pressure structure.
Both deposited cells are compressed ones, which is the single most important
fact about this system for anyone about to compute a cohesive energy from them.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*crystal system.*")

from mmml.analysis.dcm_crystal import (  # noqa: E402
    DCM_CRYSTAL_PHASES,
    KAWAGUCHI_AMBIENT_CELL,
    read_dcm_phase,
)

CELL_TOLERANCE_A = 1e-3
VOLUME_TOLERANCE_FRAC = 2e-3

FAIL: list[str] = []

print("=== 01: the deposited structures ===")
print()
print("Podsiadlo, Dziubek & Katrusiak, Acta Crystallogr. B 61, 595 (2005)")
print("(doi:10.1107/S0108768105017374), CCDC doi:10.5517/cc9lyjb, via COD.")
print()

for key, phase in DCM_CRYSTAL_PHASES.items():
    atoms = read_dcm_phase(key)
    lengths = atoms.cell.lengths()
    volume = atoms.get_volume()

    print(f"{key}  (COD {phase.cod_id})")
    print(f"  {phase.label}")
    print(f"  conditions   T = {phase.temperature_K:g} K, p = {phase.pressure_GPa:g} GPa")
    print(
        f"  cell         a={lengths[0]:.4f} b={lengths[1]:.4f} c={lengths[2]:.4f} A"
        f"   (paper: {phase.cell_lengths_A[0]:.4f} {phase.cell_lengths_A[1]:.4f} "
        f"{phase.cell_lengths_A[2]:.4f})"
    )
    print(f"  volume       {volume:.2f} A^3   (paper: {phase.cell_volume_A3:.2f})")

    for axis, built, published in zip("abc", lengths, phase.cell_lengths_A):
        if abs(built - published) > CELL_TOLERANCE_A:
            FAIL.append(f"{key}: {axis} = {built:.4f} A, paper says {published:.4f} A")
    if abs(volume - phase.cell_volume_A3) / phase.cell_volume_A3 > VOLUME_TOLERANCE_FRAC:
        FAIL.append(f"{key}: volume {volume:.2f} A^3, paper says {phase.cell_volume_A3:.2f}")

    symbols = atoms.get_chemical_symbols()
    n_c = symbols.count("C")
    n_h = symbols.count("H")
    n_cl = symbols.count("Cl")
    print(f"  contents     {len(atoms)} atoms: {n_c} C, {n_h} H, {n_cl} Cl")
    if (n_c, n_h, n_cl) != (phase.z, 2 * phase.z, 2 * phase.z):
        FAIL.append(
            f"{key}: expected Z={phase.z} CH2Cl2, got {n_c} C / {n_h} H / {n_cl} Cl"
        )

    masses = atoms.get_masses().sum()
    density = masses / volume / 0.6022140857
    print(f"  density      {density:.4f} g/cm^3   (paper: {phase.density_g_cm3:.3f})")
    if abs(density - phase.density_g_cm3) > 5e-3:
        FAIL.append(f"{key}: density {density:.4f}, paper says {phase.density_g_cm3:.3f}")

    print(f"  {phase.note}")
    print()

# --- the structure that is missing -------------------------------------------
print("-- the ambient-pressure structure, and why it is not here --")
ref = KAWAGUCHI_AMBIENT_CELL
a, b, c = ref.cell_lengths_A
print(f"  {ref.label}")
print(f"  cell         a={a:.3f} b={b:.3f} c={c:.3f} A, V={ref.cell_volume_A3:.1f} A^3")
print(f"  {ref.citation}")
print(f"  {ref.note}")
print()
print("  Both deposited structures are therefore compressed ones. Against the")
print("  ambient-pressure cell above:")
for key, phase in DCM_CRYSTAL_PHASES.items():
    ratio = phase.cell_volume_A3 / ref.cell_volume_A3
    print(
        f"      {key:12s} p = {phase.pressure_GPa:.2f} GPa   "
        f"V/V0 = {ratio:.3f}  ({100.0 * (1.0 - ratio):.1f}% compressed)"
    )
print()
print("  A crystal squeezed 11% below its ambient volume sits well up its")
print("  repulsive wall, so its static lattice energy is not a cohesive energy")
print("  and must not be compared with a sublimation enthalpy as it stands.")
print("  Step 05 relaxes the cell to zero pressure before making that")
print("  comparison, and checks the relaxed cell against the 1973 measurement.")

print()
if FAIL:
    for f in FAIL:
        print(f"ERROR:   {f}", file=sys.stderr)
    print("\n01: FAILED", file=sys.stderr)
    sys.exit(1)
print("01: OK — both cells match the published values")
