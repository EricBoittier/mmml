#!/usr/bin/env python3
"""Solvated φ/ψ scan seeded from a gas-phase trialanine PES NPZ.

For each gas-grid conformation:
  1. Build TRIA + TIP3 box with that peptide geometry (Packmol waters)
  2. CHARMM constrained MM minimize (CONS DIHE on central φ/ψ; no COM recenter)
  3. Record solvent MM energy and achieved dihedrals

Pair with ``scripts/scan_trialanine_phi_psi_pes.py`` (gas) and
``scripts/plot_tria_phi_psi_gas_solvent.py`` (figure).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import set_charmm_positions
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    ensure_pycharmm_loaded,
    pycharmm,
)
from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
    build_trialanine_water_box_in_charmm,
    n_peptide_atoms_in_trialanine_box,
)

# Match scripts/scan_trialanine_phi_psi_pes.py (central ALA of TRIA).
PHI_CENTRAL = (14, 16, 18, 24)  # C1-N2-CA2-C2
PSI_CENTRAL = (16, 18, 24, 26)  # N2-CA2-C2-N3


def _bynum(atom_indices: tuple[int, int, int, int]) -> str:
    return " ".join(str(i + 1) for i in atom_indices)


def minimize_solvated_with_phi_psi(
    *,
    phi_deg: float,
    psi_deg: float,
    force_kcal: float,
    sd_steps: int,
    abnr_steps: int,
    water_only_sd_steps: int,
) -> tuple[np.ndarray, float]:
    """Constrained MM minimize without COM-centering (keeps PBC box)."""
    import pycharmm.coor as coor
    import pycharmm.energy as energy

    commands: list[str] = ["CONS CLDH"]
    if water_only_sd_steps > 0:
        commands.extend(
            [
                "CONS FIX SELE SEGID PEPT END",
                f"MINI SD NSTEP {int(water_only_sd_steps)}",
                "CONS CLEAR",
            ]
        )
    commands.extend(
        [
            (
                f"CONS DIHE BYNUM {_bynum(PHI_CENTRAL)} FORCE {force_kcal:.8g} "
                f"MIN {float(phi_deg):.8g} PERI 0"
            ),
            (
                f"CONS DIHE BYNUM {_bynum(PSI_CENTRAL)} FORCE {force_kcal:.8g} "
                f"MIN {float(psi_deg):.8g} PERI 0"
            ),
        ]
    )
    if sd_steps > 0:
        commands.append(f"MINI SD NSTEP {int(sd_steps)}")
    if abnr_steps > 0:
        commands.append(f"MINI ABNR NSTEP {int(abnr_steps)}")
    commands.append("ENER")
    try:
        for cmd in commands:
            pycharmm.lingo.charmm_script(cmd)
        pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
        e_tot = float(energy.get_total())
        return pos, e_tot
    finally:
        pycharmm.lingo.charmm_script("CONS CLDH")
        pycharmm.lingo.charmm_script("CONS CLEAR")


def _dihedral_deg(positions: np.ndarray, idx: tuple[int, int, int, int]) -> float:
    from ase import Atoms

    # Z unused; ASE only needs 4 atoms for get_dihedral via full Atoms.
    atoms = Atoms(numbers=np.ones(len(positions), dtype=int), positions=positions)
    return float(atoms.get_dihedral(*idx))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gas-npz",
        type=Path,
        required=True,
        help="Gas-phase phi_psi_pes.npz from scan_trialanine_phi_psi_pes.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/tria_phi_psi_scan/solvent"),
    )
    parser.add_argument("--n-waters", type=int, default=50)
    parser.add_argument("--box-side-A", type=float, default=28.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mm-sd-steps", type=int, default=100)
    parser.add_argument("--mm-abnr-steps", type=int, default=100)
    parser.add_argument(
        "--water-only-sd-steps",
        type=int,
        default=50,
        help="SD steps with PEPT fixed before joint constrained mini (0 to skip)",
    )
    parser.add_argument(
        "--mm-dihedral-force",
        type=float,
        default=500.0,
        help="CHARMM CONS DIHE force (kcal/mol/rad^2)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap for smoke tests (first N grid points in C-order)",
    )
    args = parser.parse_args()

    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

    gas = np.load(args.gas_npz)
    phi_grid = np.asarray(gas["phi_grid_deg"], dtype=float)
    psi_grid = np.asarray(gas["psi_grid_deg"], dtype=float)
    positions_A = np.asarray(gas["positions_A"], dtype=float)
    gas_mm = np.asarray(gas["charmm_mm_min_energy_kcal_mol"], dtype=float)
    gas_ml = np.asarray(gas.get("ml_energy_eV", np.full_like(gas_mm, np.nan)), dtype=float)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "boxes"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    rows: list[dict[str, float | str | bool]] = []
    n_phi, n_psi = len(phi_grid), len(psi_grid)
    n_done = 0
    for i, phi in enumerate(phi_grid):
        for j, psi in enumerate(psi_grid):
            if args.max_points is not None and n_done >= int(args.max_points):
                break
            pept = positions_A[i, j]
            if not np.all(np.isfinite(pept)):
                print(f"skip phi={phi:.1f} psi={psi:.1f}: non-finite gas positions", flush=True)
                continue
            tag = f"phi_{phi:+07.2f}_psi_{psi:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
            box_dir = work_root / tag
            box_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{n_done + 1}] solvent relax phi={phi:7.2f} psi={psi:7.2f} …",
                flush=True,
            )
            box = build_trialanine_water_box_in_charmm(
                n_waters=int(args.n_waters),
                box_side_A=float(args.box_side_A),
                seed=int(args.seed) + n_done,
                workdir=box_dir,
                peptide_positions=pept,
            )
            n_pept = n_peptide_atoms_in_trialanine_box(box.psf_path)
            if pept.shape[0] != n_pept:
                raise RuntimeError(
                    f"gas peptide atoms {pept.shape[0]} != box PEPT atoms {n_pept}"
                )
            # Re-assert peptide coords after Packmol (waters may have shifted COM).
            all_pos = np.asarray(box.positions, dtype=float).copy()
            pept_boxed = pept - pept.mean(axis=0) + np.array(
                [args.box_side_A / 2] * 3, dtype=float
            )
            all_pos[:n_pept] = pept_boxed
            set_charmm_positions(all_pos)

            final_pos, e_sol = minimize_solvated_with_phi_psi(
                phi_deg=float(phi),
                psi_deg=float(psi),
                force_kcal=float(args.mm_dihedral_force),
                sd_steps=int(args.mm_sd_steps),
                abnr_steps=int(args.mm_abnr_steps),
                water_only_sd_steps=int(args.water_only_sd_steps),
            )
            phi_act = _dihedral_deg(final_pos[:n_pept], PHI_CENTRAL)
            psi_act = _dihedral_deg(final_pos[:n_pept], PSI_CENTRAL)
            np.savez(
                box_dir / "relaxed.npz",
                positions=final_pos,
                energy_kcal_mol=e_sol,
                phi_deg=float(phi),
                psi_deg=float(psi),
                actual_phi_deg=phi_act,
                actual_psi_deg=psi_act,
            )
            rows.append(
                {
                    "phi_deg": float(phi),
                    "psi_deg": float(psi),
                    "actual_phi_deg": phi_act,
                    "actual_psi_deg": psi_act,
                    "gas_charmm_mm_min_energy_kcal_mol": float(gas_mm[i, j]),
                    "gas_ml_energy_eV": float(gas_ml[i, j]),
                    "solvent_mm_min_energy_kcal_mol": float(e_sol),
                    "n_waters": int(args.n_waters),
                    "box_side_A": float(args.box_side_A),
                    "box_dir": str(box_dir),
                }
            )
            print(
                f"    E_sol={e_sol:12.4f} kcal/mol  "
                f"φ/ψ → {phi_act:7.2f}/{psi_act:7.2f}",
                flush=True,
            )
            n_done += 1
        if args.max_points is not None and n_done >= int(args.max_points):
            break

    if not rows:
        raise RuntimeError("No solvent grid points completed")

    csv_path = out_dir / "phi_psi_solvent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_json(out_dir / "phi_psi_solvent.json", orient="records", indent=2)
    print(f"Wrote {csv_path} ({len(rows)} points)", flush=True)


if __name__ == "__main__":
    main()
