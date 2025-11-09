#!/usr/bin/env python3
"""
Compute dipole moments for trajectory frames produced by jaxmd_dynamics.py.

The script expects a positions NPZ (e.g. multi_copy_traj_*.npz) and a metadata
NPZ containing atomic_numbers (written alongside the trajectory). Dipoles are
evaluated with the saved JointPhysNetDCMNet checkpoint and stored in a new NPZ.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path to import trainer
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "../../.."
sys.path.insert(0, str(REPO_ROOT.resolve()))

from trainer import JointPhysNetDCMNet  # noqa: E402


def load_checkpoint(checkpoint_dir: Path) -> Tuple[JointPhysNetDCMNet, dict]:
    with open(checkpoint_dir / "best_params.pkl", "rb") as f:
        params = pickle.load(f)
    with open(checkpoint_dir / "model_config.pkl", "rb") as f:
        config = pickle.load(f)
    model = JointPhysNetDCMNet(
        physnet_config=config["physnet_config"],
        dcmnet_config=config["dcmnet_config"],
        mix_coulomb_energy=config.get("mix_coulomb_energy", False),
    )
    return model, params


def build_neighbor_graph(positions: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    dst_list, src_list = [], []
    n_atoms = positions.shape[0]
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            if np.linalg.norm(positions[i] - positions[j]) < cutoff:
                dst_list.append(i)
                src_list.append(j)
    if not dst_list:
        dst_list = [0]
        src_list = [0]
    return np.array(dst_list, dtype=np.int32), np.array(src_list, dtype=np.int32), len(dst_list) == 1 and dst_list[0] == 0 and src_list[0] == 0


def evaluate_single(
    model: JointPhysNetDCMNet,
    params: dict,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cutoff: float,
    model_natoms: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_atoms = atomic_numbers.shape[0]
    positions_pad = np.zeros((model_natoms, 3), dtype=np.float32)
    positions_pad[:n_atoms] = positions.astype(np.float32)

    atomic_numbers_pad = np.zeros((model_natoms,), dtype=np.int32)
    atomic_numbers_pad[:n_atoms] = atomic_numbers.astype(np.int32)

    atom_mask = np.zeros((model_natoms,), dtype=np.float32)
    atom_mask[:n_atoms] = 1.0

    dst_idx_np, src_idx_np, empty_edges = build_neighbor_graph(positions[:n_atoms], cutoff)
    batch_mask = np.zeros((len(dst_idx_np),), dtype=np.float32) if empty_edges else np.ones((len(dst_idx_np),), dtype=np.float32)

    output = model.apply(
        params,
        atomic_numbers=jnp.array(atomic_numbers_pad),
        positions=jnp.array(positions_pad),
        dst_idx=jnp.array(dst_idx_np),
        src_idx=jnp.array(src_idx_np),
        batch_segments=jnp.zeros((model_natoms,), dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.array(batch_mask),
        atom_mask=jnp.array(atom_mask),
    )

    dipole_physnet = np.array(output.get("dipoles", output.get("dipoles_mixed"))[0])
    dipole_dcmnet = np.array(output.get("dipoles_dcmnet", output.get("dipoles", np.zeros(3)))[0])
    energy = float(output["energy"][0])
    return dipole_physnet, dipole_dcmnet, energy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dipole moments for stored trajectories.")
    parser.add_argument("--positions", type=Path, required=True, help="NPZ file with positions array")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata NPZ with atomic_numbers")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ for dipoles")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Neighbor cutoff (Å)")
    args = parser.parse_args()

    positions_npz = np.load(args.positions)
    positions = positions_npz["positions"]
    positions_npz.close()

    meta = np.load(args.metadata)
    atomic_numbers = meta["atomic_numbers"]
    masses = meta.get("masses")
    temperature = meta.get("temperature")
    meta.close()

    model, params = load_checkpoint(args.checkpoint)
    model_natoms = model.physnet_config["natoms"]

    if positions.ndim == 4:
        n_steps, n_replica, n_atoms, _ = positions.shape
    elif positions.ndim == 3:
        n_steps, n_atoms = positions.shape[0], positions.shape[1]
        n_replica = 1
        positions = positions[:, None, :, :]
    else:
        raise ValueError("Unsupported positions array shape.")

    print(f"Frames: {n_steps}, replicas: {n_replica}, atoms: {n_atoms}")

    dipoles_physnet = np.zeros((n_steps, n_replica, 3), dtype=np.float32)
    dipoles_dcmnet = np.zeros((n_steps, n_replica, 3), dtype=np.float32)
    energies = np.zeros((n_steps, n_replica), dtype=np.float32)

    for step in range(n_steps):
        for replica in range(n_replica):
            dip_phys, dip_dcm, energy = evaluate_single(
                model,
                params,
                positions[step, replica],
                atomic_numbers,
                args.cutoff,
                model_natoms,
            )
            dipoles_physnet[step, replica] = dip_phys
            dipoles_dcmnet[step, replica] = dip_dcm
            energies[step, replica] = energy
        if (step + 1) % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"Processed {step + 1}/{n_steps} steps")

    output_path = args.output or args.positions.with_name(args.positions.stem + "_dipoles.npz")
    np.savez(
        output_path,
        dipoles_physnet=dipoles_physnet,
        dipoles_dcmnet=dipoles_dcmnet,
        energies=energies,
        atomic_numbers=atomic_numbers,
        masses=masses,
        temperature=temperature,
        cutoff=args.cutoff,
    )
    print(f"✅ Saved dipoles to {output_path}")


if __name__ == "__main__":
    main()

