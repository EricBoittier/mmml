#!/usr/bin/env python3
"""
Simple NumPy-based utilities for inspecting trajectories saved by
`jaxmd_dynamics.py --multi-replicas`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_trajectory(path: Path) -> np.ndarray:
    data = np.load(path)
    if "positions" not in data:
        raise KeyError(f"'positions' not found in {path}")
    return data["positions"]


def rms_displacement(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = traj[0]
    displacements = traj - reference
    rms_per_replica = np.sqrt(np.mean(np.sum(displacements**2, axis=-1), axis=0))  # shape (replicas,)
    rms_per_step = np.sqrt(np.mean(np.sum(displacements**2, axis=-1), axis=(1, 2)))  # shape (steps,)
    return rms_per_replica, rms_per_step


def bond_lengths(traj: np.ndarray, replica: int = 0) -> np.ndarray:
    positions = traj[:, replica, :, :]
    n_atoms = positions.shape[1]
    lengths = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            diff = positions[:, i] - positions[:, j]
            lengths.append(np.linalg.norm(diff, axis=-1))
    return np.stack(lengths, axis=-1)


def summarize(traj: np.ndarray, replica_for_bonds: int = 0) -> Dict[str, np.ndarray]:
    pos_min = traj.min(axis=(0, 1, 2))
    pos_max = traj.max(axis=(0, 1, 2))
    rms_replica, rms_step = rms_displacement(traj)
    bonds = bond_lengths(traj, replica_for_bonds)
    return {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "rms_per_replica": rms_replica,
        "rms_per_step": rms_step,
        "bond_lengths": bonds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-replica trajectories (NumPy only).")
    parser.add_argument("trajectory", type=Path, help="NPZ file produced by jaxmd_dynamics.py")
    parser.add_argument("--replica", type=int, default=0, help="Replica index for bond-length reporting")
    parser.add_argument("--save-json", type=Path, help="Optional path to dump summary JSON")
    parser.add_argument("--save-csv", type=Path, help="Optional CSV path for bond-length time series")
    args = parser.parse_args()

    traj = load_trajectory(args.trajectory)
    summary = summarize(traj, args.replica)

    n_steps, n_replica, n_atoms, _ = traj.shape
    print(f"Trajectory: {args.trajectory}")
    print(f"  Steps: {n_steps}")
    print(f"  Replicas: {n_replica}")
    print(f"  Atoms: {n_atoms}")
    pos_min = float(summary["pos_min"].min())
    pos_max = float(summary["pos_max"].max())
    print(f"  Position range: [{pos_min:.6f}, {pos_max:.6f}] Å")
    rms_replica = summary["rms_per_replica"]
    print(f"  RMS displacement per replica (Å):")
    for idx in range(rms_replica.shape[0]):
        scalar = float(np.asarray(rms_replica[idx]).squeeze())
        print(f"    replica {idx:>3}: {scalar:.6e}")
    rms_step = summary["rms_per_step"]
    rms_step_min = float(np.asarray(rms_step.min()).squeeze())
    rms_step_max = float(np.asarray(rms_step.max()).squeeze())
    print(f"  RMS displacement per step (Å): min={rms_step_min:.6e}, "
          f"max={rms_step_max:.6e}")

    bonds = summary["bond_lengths"]
    print(f"  Bond lengths for replica {args.replica}:")
    for bond_idx in range(bonds.shape[1]):
        series = bonds[:, bond_idx]
        print(
            f"    bond {bond_idx:>2} -> "
            f"mean={series.mean():.6f} Å, std={series.std():.6f}, "
            f"min={series.min():.6f}, max={series.max():.6f}"
        )

    if args.save_json:
        serializable = {
            "pos_min": summary["pos_min"].tolist(),
            "pos_max": summary["pos_max"].tolist(),
            "rms_per_replica": summary["rms_per_replica"].tolist(),
            "rms_per_step": summary["rms_per_step"].tolist(),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  JSON summary saved to {args.save_json}")

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join([f"bond_{i}" for i in range(bonds.shape[1])])
        np.savetxt(args.save_csv, bonds, delimiter=",", header=header, comments="")
        print(f"  Bond length series saved to {args.save_csv}")


if __name__ == "__main__":
    main()

