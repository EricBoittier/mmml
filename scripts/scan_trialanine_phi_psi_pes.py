#!/usr/bin/env python3
"""Scan central tri-alanine PHI/PSI and evaluate CHARMM plus ML ASE PES grids."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixInternals
from ase.io import write
from ase.optimize.fire import FIRE

from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.interfaces.pycharmmInterface import setupRes
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import set_charmm_positions
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    crystal_free_charmm_for_param_append,
    ensure_pycharmm_loaded,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill
from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
    TRIA_RESI_NAME,
    _load_cgenff_with_trialanine,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf


PEPTIDE_CKPT_PATH = "/mmhome/boittier/home/mmml_tutorial/aaa.ama/aaa_model/ckpts/test01/params_test01_2026-07-08_12-34-45.json"

PHI_CENTRAL = (14, 16, 18, 24)  # C1-N2-CA2-C2
PSI_CENTRAL = (16, 18, 24, 26)  # N2-CA2-C2-N3


def build_trialanine_peptide_in_charmm() -> tuple[np.ndarray, np.ndarray]:
    """Build PEPT/TRIA in CHARMM and return atomic numbers plus coordinates."""
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings

    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is not available; set CHARMM_LIB_DIR/libcharmm first.")

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)

    read.sequence_string(TRIA_RESI_NAME)
    generate.new_segment(seg_name="PEPT", setup_ic=True)
    ic_prm_fill(replace_all=True)
    ic.build()

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    if np.any(np.abs(positions) > 9000.0) or float(np.std(positions)) < 0.05:
        setupRes.generate_coordinates(skip_energy_show=True, validate=True)
        positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)

    positions = positions - positions.mean(axis=0)
    set_charmm_positions(positions)
    atomic_numbers = np.asarray(get_Z_from_psf(), dtype=np.int32)
    return atomic_numbers, positions


def set_phi_psi(atoms: Atoms, phi_deg: float, psi_deg: float) -> Atoms:
    """Return a copy with central PHI/PSI set by ASE rotations."""
    scanned = atoms.copy()
    n_atoms = len(scanned)

    phi_mask = np.zeros(n_atoms, dtype=bool)
    phi_mask[PHI_CENTRAL[2] :] = True
    scanned.set_dihedral(*PHI_CENTRAL, phi_deg, mask=phi_mask)

    psi_mask = np.zeros(n_atoms, dtype=bool)
    psi_mask[PSI_CENTRAL[2] :] = True
    scanned.set_dihedral(*PSI_CENTRAL, psi_deg, mask=psi_mask)
    return scanned


def relax_with_fixed_phi_psi(
    atoms: Atoms,
    calc,
    phi_deg: float,
    psi_deg: float,
    *,
    fmax: float,
    steps: int,
) -> Atoms:
    """Relax all other coordinates with PHI/PSI fixed."""
    relaxed = atoms.copy()
    relaxed.calc = calc
    relaxed.set_constraint(
        FixInternals(
            dihedrals_deg=[
                [float(phi_deg), list(PHI_CENTRAL)],
                [float(psi_deg), list(PSI_CENTRAL)],
            ]
        )
    )
    opt = FIRE(relaxed, logfile=None, maxstep=0.03)
    opt.run(fmax=fmax, steps=steps)
    relaxed.set_constraint()
    return relaxed


def charmm_energy_kcal(positions: np.ndarray) -> float:
    """Evaluate CHARMM MM energy at the supplied coordinates."""
    import pycharmm.energy as energy

    set_charmm_positions(np.asarray(positions, dtype=np.float64))
    pycharmm.lingo.charmm_script("ENER")
    return float(energy.get_total())


def parse_grid(values: str) -> np.ndarray:
    """Parse start:stop:step grid in degrees, including stop."""
    start, stop, step = (float(x) for x in values.split(":"))
    if step <= 0:
        raise ValueError("grid step must be positive")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=PEPTIDE_CKPT_PATH)
    parser.add_argument("--phi", default="-180:180:30", help="PHI grid as start:stop:step degrees")
    parser.add_argument("--psi", default="-180:180:30", help="PSI grid as start:stop:step degrees")
    parser.add_argument("--out", default="artifacts/trialanine_phi_psi_pes")
    parser.add_argument("--relax-ase", action="store_true", help="Constrained ML relaxation before energy evaluation")
    parser.add_argument("--relax-steps", type=int, default=200)
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_numbers, positions = build_trialanine_peptide_in_charmm()
    base_atoms = Atoms(numbers=atomic_numbers, positions=positions)

    calc = create_calculator_from_checkpoint(args.checkpoint)
    phi_grid = parse_grid(args.phi)
    psi_grid = parse_grid(args.psi)

    charmm_kcal = np.full((len(phi_grid), len(psi_grid)), np.nan, dtype=float)
    ml_ev = np.full_like(charmm_kcal, np.nan)
    actual_phi = np.full_like(charmm_kcal, np.nan)
    actual_psi = np.full_like(charmm_kcal, np.nan)

    rows: list[dict[str, float]] = []
    for i, phi in enumerate(phi_grid):
        for j, psi in enumerate(psi_grid):
            atoms = set_phi_psi(base_atoms, phi, psi)
            if args.relax_ase:
                atoms = relax_with_fixed_phi_psi(
                    atoms,
                    calc,
                    phi,
                    psi,
                    fmax=args.relax_fmax,
                    steps=args.relax_steps,
                )
            atoms.calc = calc

            e_ml = float(atoms.get_potential_energy())
            e_charmm = charmm_energy_kcal(atoms.get_positions())
            phi_actual = float(atoms.get_dihedral(*PHI_CENTRAL))
            psi_actual = float(atoms.get_dihedral(*PSI_CENTRAL))

            ml_ev[i, j] = e_ml
            charmm_kcal[i, j] = e_charmm
            actual_phi[i, j] = phi_actual
            actual_psi[i, j] = psi_actual
            rows.append(
                {
                    "phi_deg": float(phi),
                    "psi_deg": float(psi),
                    "actual_phi_deg": phi_actual,
                    "actual_psi_deg": psi_actual,
                    "ml_energy_eV": e_ml,
                    "charmm_energy_kcal_mol": e_charmm,
                }
            )
            print(
                f"phi={phi:7.2f} psi={psi:7.2f} "
                f"ML={e_ml:14.6f} eV CHARMM={e_charmm:14.6f} kcal/mol",
                flush=True,
            )

            if args.write_xyz:
                write(out_dir / f"phi_{phi:+07.2f}_psi_{psi:+07.2f}.xyz", atoms)

    np.savez(
        out_dir / "phi_psi_pes.npz",
        phi_grid_deg=phi_grid,
        psi_grid_deg=psi_grid,
        ml_energy_eV=ml_ev,
        charmm_energy_kcal_mol=charmm_kcal,
        actual_phi_deg=actual_phi,
        actual_psi_deg=actual_psi,
        phi_atoms=np.asarray(PHI_CENTRAL, dtype=np.int32),
        psi_atoms=np.asarray(PSI_CENTRAL, dtype=np.int32),
    )

    csv_path = out_dir / "phi_psi_pes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pd.DataFrame(rows).to_json(out_dir / "phi_psi_pes.json", orient="records", indent=2)
    print(f"Wrote {out_dir / 'phi_psi_pes.npz'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
