#!/usr/bin/env python3
"""Solvated φ/ψ scan seeded from a gas-phase trialanine PES NPZ.

Builds the TRIA + TIP3 box **once**, then for each gas-grid conformation:
  1. Inject peptide coordinates (waters restored from the packed reference)
  2. CHARMM constrained MM minimize (CONS DIHE on central φ/ψ; no COM recenter)
  3. Record solvent MM energy and achieved dihedrals

Rebuilding Packmol/CHARMM every grid point tends to abort silently in libcharmm
after a handful of DELETE cycles — keep one live system instead.

Pair with ``scripts/scan_trialanine_phi_psi_pes.py`` (gas) and
``scripts/plot_tria_phi_psi_gas_solvent.py`` (figure).
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import shutil
import sys
import traceback
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

faulthandler.enable()


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

    commands: list[str] = ["CONS CLDH", "CONS CLEAR"]
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

    # ASE returns [0, 360); wrap to (−180, 180] for Ramachandran / CONS DIHE.
    atoms = Atoms(numbers=np.ones(len(positions), dtype=int), positions=positions)
    ang = float(atoms.get_dihedral(*idx))
    return ((ang + 180.0) % 360.0) - 180.0


def _place_peptide(pept: np.ndarray, box_side_A: float) -> np.ndarray:
    center = np.array([box_side_A / 2, box_side_A / 2, box_side_A / 2], dtype=float)
    return np.asarray(pept, dtype=float) - pept.mean(axis=0) + center


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

    # Seed Packmol with the first finite gas frame.
    seed_pept = None
    for i0 in range(positions_A.shape[0]):
        for j0 in range(positions_A.shape[1]):
            if np.all(np.isfinite(positions_A[i0, j0])):
                seed_pept = positions_A[i0, j0]
                break
        if seed_pept is not None:
            break
    if seed_pept is None:
        raise RuntimeError("No finite peptide frames in gas NPZ")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "boxes"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    box_dir = work_root / "_shared_box"
    box_dir.mkdir(parents=True)

    print(
        f"Building shared solvent box once "
        f"(n_waters={args.n_waters}, L={args.box_side_A} Å) …",
        flush=True,
    )
    box = build_trialanine_water_box_in_charmm(
        n_waters=int(args.n_waters),
        box_side_A=float(args.box_side_A),
        seed=int(args.seed),
        workdir=box_dir,
        peptide_positions=seed_pept,
    )
    n_pept = n_peptide_atoms_in_trialanine_box(box.psf_path)
    if seed_pept.shape[0] != n_pept:
        raise RuntimeError(
            f"gas peptide atoms {seed_pept.shape[0]} != box PEPT atoms {n_pept}"
        )
    ref_pos = np.asarray(box.positions, dtype=float).copy()
    print(
        f"Shared box ready: {ref_pos.shape[0]} atoms "
        f"({n_pept} PEPT + {args.n_waters} TIP3)",
        flush=True,
    )

    rows: list[dict[str, float | str | bool]] = []
    csv_path = out_dir / "phi_psi_solvent.csv"
    n_done = 0
    n_fail = 0
    for i, phi in enumerate(phi_grid):
        for j, psi in enumerate(psi_grid):
            if args.max_points is not None and n_done + n_fail >= int(args.max_points):
                break
            pept = positions_A[i, j]
            if not np.all(np.isfinite(pept)):
                print(
                    f"skip phi={phi:.1f} psi={psi:.1f}: non-finite gas positions",
                    flush=True,
                )
                continue
            tag = (
                f"phi_{phi:+07.2f}_psi_{psi:+07.2f}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )
            point_dir = work_root / tag
            point_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{n_done + n_fail + 1}] solvent relax "
                f"phi={phi:7.2f} psi={psi:7.2f} …",
                flush=True,
            )
            try:
                # Restore packed waters; only peptide changes between points.
                all_pos = ref_pos.copy()
                all_pos[:n_pept] = _place_peptide(pept, float(args.box_side_A))
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
                    point_dir / "relaxed.npz",
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
                        "box_dir": str(point_dir),
                        "ok": True,
                    }
                )
                print(
                    f"    E_sol={e_sol:12.4f} kcal/mol  "
                    f"φ/ψ → {phi_act:7.2f}/{psi_act:7.2f}",
                    flush=True,
                )
                n_done += 1
            except Exception as exc:  # noqa: BLE001 — keep grid going
                n_fail += 1
                print(
                    f"    FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                rows.append(
                    {
                        "phi_deg": float(phi),
                        "psi_deg": float(psi),
                        "actual_phi_deg": float("nan"),
                        "actual_psi_deg": float("nan"),
                        "gas_charmm_mm_min_energy_kcal_mol": float(gas_mm[i, j]),
                        "gas_ml_energy_eV": float(gas_ml[i, j]),
                        "solvent_mm_min_energy_kcal_mol": float("nan"),
                        "n_waters": int(args.n_waters),
                        "box_side_A": float(args.box_side_A),
                        "box_dir": str(point_dir),
                        "ok": False,
                    }
                )
            _write_rows(csv_path, rows)
        if args.max_points is not None and n_done + n_fail >= int(args.max_points):
            break

    if not any(r.get("ok") for r in rows):
        raise RuntimeError("No solvent grid points completed successfully")

    pd.DataFrame(rows).to_json(out_dir / "phi_psi_solvent.json", orient="records", indent=2)
    print(
        f"Wrote {csv_path} ({n_done} ok, {n_fail} failed)",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
