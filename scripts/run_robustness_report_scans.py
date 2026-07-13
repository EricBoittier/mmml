#!/usr/bin/env python3
"""Real 1D bond-stretch and angle-bend potential-energy scans on a single
water molecule, using the same real charge-predicting PhysNet checkpoint as
`scripts/run_robustness_report_md.py`. Complements the dihedral/torsion
scan already on disk (`artifacts/trialanine_phi_psi_mm_then_ml_64x64/
phi_psi_pes.csv`) and the dimer-separation scan
(`results/dimer_scan_campaign/scan_results.csv`) so the robustness report
covers all three internal-coordinate scan types (bond/angle/dihedral) from
real model evaluations, none synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from mmml.interfaces.calculators.checkpoint_loading import create_calculator_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = REPO_ROOT / "mmml/models/physnetjax/defaults/hf_json/test-b4064dca-8cbd-471c-9871-08887107a1d8_epoch-550_portable.json"
OUT_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "scans"

EQ_BOND_A = 0.9572
EQ_ANGLE_DEG = 104.5


def _water(bond_A: float, angle_deg: float) -> Atoms:
    half = np.deg2rad(angle_deg / 2)
    o = np.zeros(3)
    h1 = o + bond_A * np.array([np.cos(half), np.sin(half), 0.0])
    h2 = o + bond_A * np.array([np.cos(half), -np.sin(half), 0.0])
    return Atoms(numbers=[8, 1, 1], positions=[o, h1, h2])


def bond_scan(calc) -> None:
    bonds = np.linspace(0.75, 1.35, 41)  # A -- covers well past both turning points
    energies = np.zeros_like(bonds)
    for i, b in enumerate(bonds):
        atoms = _water(b, EQ_ANGLE_DEG)
        atoms.calc = calc
        energies[i] = atoms.get_potential_energy()
    np.savez(OUT_DIR / "bond_scan.npz", bond_A=bonds, energy_eV=energies,
             eq_bond_A=EQ_BOND_A, checkpoint=str(CKPT.relative_to(REPO_ROOT)))
    print(f"wrote {OUT_DIR / 'bond_scan.npz'} (min at {bonds[np.argmin(energies)]:.3f} A)")


def angle_scan(calc) -> None:
    angles = np.linspace(80.0, 140.0, 41)  # degrees
    energies = np.zeros_like(angles)
    for i, a in enumerate(angles):
        atoms = _water(EQ_BOND_A, a)
        atoms.calc = calc
        energies[i] = atoms.get_potential_energy()
    np.savez(OUT_DIR / "angle_scan.npz", angle_deg=angles, energy_eV=energies,
             eq_angle_deg=EQ_ANGLE_DEG, checkpoint=str(CKPT.relative_to(REPO_ROOT)))
    print(f"wrote {OUT_DIR / 'angle_scan.npz'} (min at {angles[np.argmin(energies)]:.1f} deg)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calc = create_calculator_from_checkpoint(str(CKPT))
    bond_scan(calc)
    angle_scan(calc)


if __name__ == "__main__":
    main()
