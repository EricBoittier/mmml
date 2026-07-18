#!/usr/bin/env python3
"""Sample structures from SO3LR/Spooky extxyz test sets as true-vs-predicted ASE trajectories.

For each --extxyz file, samples --num-samples random structures and writes
TWO .traj files per dataset: one with the reference (true) energy/forces
attached, one with SpookyNetCalculator's predicted energy/forces attached
for the SAME structures in the SAME order -- so they can be diffed frame by
frame (`ase gui foo_true.traj foo_pred.traj`, or loaded in a notebook and
compared programmatically).

Uses mmml.models.spookynet_calc.SpookyNetCalculator directly (no CHARMM, no
hybrid ML/MM decomposition, no PBC) -- the standalone single-molecule path.
Accepts either a raw orbax epoch-N checkpoint directory or a portable JSON.

Usage:
    python scripts/sample_so3lr_eval_trajectories.py \\
        --checkpoint /mmhome/boittier/home/mmml/artifacts/spooky_so3lr_muon3/epoch-0010 \\
        --extxyz ~/data/so3lr_test/ \\
        --exclude-substring gems \\
        --num-samples 5 \\
        --out-dir eval_out/sample_trajectories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread
from ase.io.trajectory import Trajectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _infer_spin_multiplicity(atoms, total_charge: float) -> float:
    """Mirrors evaluate_so3lr_spooky_extxyz.py's singlet/doublet inference."""
    n_protons = int(np.sum(atoms.get_atomic_numbers()))
    n_electrons = int(round(n_protons - total_charge))
    return 1.0 if n_electrons % 2 == 0 else 2.0


def _true_energy_forces(atoms, energy_key: str, forces_key: str) -> tuple[float, np.ndarray]:
    if energy_key in atoms.info:
        energy = float(np.asarray(atoms.info[energy_key]).reshape(-1)[0])
    elif atoms.calc is not None and energy_key in getattr(atoms.calc, "results", {}):
        energy = float(np.asarray(atoms.calc.results[energy_key]).reshape(-1)[0])
    else:
        energy = float(atoms.get_potential_energy())
    if forces_key in atoms.arrays:
        forces = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
    elif atoms.calc is not None and forces_key in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results[forces_key], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return energy, forces


def _resolve_extxyz_files(extxyz_arg: Path, exclude_substring: str) -> list[Path]:
    if extxyz_arg.is_dir():
        files = sorted(extxyz_arg.glob("*.extxyz"))
    else:
        files = [extxyz_arg]
    if exclude_substring:
        files = [f for f in files if exclude_substring not in f.name]
    if not files:
        raise FileNotFoundError(f"No .extxyz files found at {extxyz_arg} (after exclusions)")
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Orbax epoch-N dir or portable JSON")
    p.add_argument("--extxyz", required=True, type=Path, help="A .extxyz file or a directory of them")
    p.add_argument("--num-samples", type=int, default=5, help="Structures sampled per dataset (default: 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("eval_out/sample_trajectories"))
    p.add_argument(
        "--exclude-substring",
        default="gems",
        help="Skip .extxyz files whose name contains this substring (default: 'gems'; pass '' to disable).",
    )
    p.add_argument("--energy-key", default="energy")
    p.add_argument("--forces-key", default="forces")
    p.add_argument("--charge-key", default="charge")
    args = p.parse_args()

    from mmml.models.spookynet_calc import SpookyNetCalculator

    files = _resolve_extxyz_files(args.extxyz, args.exclude_substring)
    print(f"Sampling {args.num_samples} structure(s) from {len(files)} dataset(s): "
          f"{', '.join(f.name for f in files)}")

    calc = SpookyNetCalculator(checkpoint=args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for extxyz_file in files:
        atoms_list = list(iread(extxyz_file, index=":"))
        rng = np.random.default_rng(args.seed)
        n_pick = min(args.num_samples, len(atoms_list))
        picks = sorted(rng.choice(len(atoms_list), size=n_pick, replace=False).tolist())

        true_path = args.out_dir / f"{extxyz_file.stem}_true.traj"
        pred_path = args.out_dir / f"{extxyz_file.stem}_pred.traj"
        true_traj = Trajectory(str(true_path), "w")
        pred_traj = Trajectory(str(pred_path), "w")

        print(f"\n--- {extxyz_file.name}: structures {picks} ---")
        print(f"{'idx':>6} {'n_atoms':>8} {'E_true':>12} {'E_pred':>12} {'dE':>10} {'|F_true|max':>12} {'|F_pred|max':>12}")

        for idx in picks:
            atoms = atoms_list[idx]
            e_true, f_true = _true_energy_forces(atoms, args.energy_key, args.forces_key)

            true_atoms = atoms.copy()
            true_atoms.calc = SinglePointCalculator(true_atoms, energy=e_true, forces=f_true)
            true_traj.write(true_atoms)

            charge = float(atoms.info.get(args.charge_key, 0.0))
            calc.charge = charge
            calc.spin_multiplicity = _infer_spin_multiplicity(atoms, charge)

            pred_atoms = atoms.copy()
            pred_atoms.calc = calc
            e_pred = float(pred_atoms.get_potential_energy())
            f_pred = np.asarray(pred_atoms.get_forces())
            pred_atoms.calc = SinglePointCalculator(pred_atoms, energy=e_pred, forces=f_pred)
            pred_traj.write(pred_atoms)

            print(
                f"{idx:>6} {len(atoms):>8} {e_true:>12.4f} {e_pred:>12.4f} "
                f"{e_pred - e_true:>10.4f} {np.abs(f_true).max():>12.4f} {np.abs(f_pred).max():>12.4f}"
            )

        true_traj.close()
        pred_traj.close()
        print(f"wrote {true_path}")
        print(f"wrote {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
