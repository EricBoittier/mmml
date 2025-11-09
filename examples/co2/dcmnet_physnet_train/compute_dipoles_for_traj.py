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
import time
from datetime import datetime, timedelta
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


def build_dense_graph(model_natoms: int) -> Tuple[np.ndarray, np.ndarray]:
    dst_list, src_list = [], []
    for i in range(model_natoms):
        for j in range(model_natoms):
            if i != j:
                dst_list.append(i)
                src_list.append(j)
    if not dst_list:
        dst_list = [0]
        src_list = [0]
    return np.array(dst_list, dtype=np.int32), np.array(src_list, dtype=np.int32)


def prepare_static_inputs(atomic_numbers: np.ndarray,
                          model_natoms: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_atoms = atomic_numbers.shape[0]
    atomic_numbers_pad = np.zeros((model_natoms,), dtype=np.int32)
    atomic_numbers_pad[:n_atoms] = atomic_numbers.astype(np.int32)
    atom_mask = np.zeros((model_natoms,), dtype=np.float32)
    atom_mask[:n_atoms] = 1.0
    return jnp.array(atomic_numbers_pad), jnp.array(atom_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dipole moments for stored trajectories.")
    parser.add_argument("--positions", type=Path, required=True, help="NPZ file with positions array")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata NPZ with atomic_numbers")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ for dipoles")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Neighbor cutoff (Å)")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of frames per evaluation batch")
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

    atomic_numbers_pad, atom_mask = prepare_static_inputs(atomic_numbers, model_natoms)
    n_total = n_steps * n_replica
    positions_flat = positions.reshape(n_total, n_atoms, 3)
    positions_pad = np.zeros((n_total, model_natoms, 3), dtype=np.float32)
    positions_pad[:, :n_atoms, :] = positions_flat.astype(np.float32)

    dst_idx_np, src_idx_np = build_dense_graph(model_natoms)
    dst_idx = jnp.array(dst_idx_np)
    src_idx = jnp.array(src_idx_np)
    batch_segments = jnp.zeros((model_natoms,), dtype=jnp.int32)
    batch_mask = jnp.ones((len(dst_idx_np),), dtype=jnp.float32)

    @jax.jit
    def single_forward(pos_pad: jnp.ndarray):
        output = model.apply(
            params,
            atomic_numbers=atomic_numbers_pad,
            positions=pos_pad,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        dipole_physnet = output.get("dipoles", output.get("dipoles_mixed"))[0]
        dipole_dcmnet = output.get("dipoles_dcmnet", output.get("dipoles"))[0]
        energy = output["energy"][0]
        return dipole_physnet, dipole_dcmnet, energy

    forward_chunk = jax.jit(lambda pos_chunk: jax.vmap(single_forward)(pos_chunk))

    dip_phys_chunks = []
    dip_dcm_chunks = []
    energy_chunks = []

    total = positions_pad.shape[0]
    batch_size = max(1, args.batch_size)
    n_batches = (total + batch_size - 1) // batch_size
    
    print(f"Processing {total} frames in {n_batches} batches of {batch_size}...")
    
    start_time = time.time()
    last_print_time = start_time
    print_interval = 5.0  # Print every 5 seconds minimum
    
    for batch_idx, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        pos_chunk = jnp.asarray(positions_pad[start:end], dtype=jnp.float32)
        dip_phys_chunk, dip_dcm_chunk, energy_chunk = forward_chunk(pos_chunk)
        dip_phys_chunks.append(np.asarray(dip_phys_chunk))
        dip_dcm_chunks.append(np.asarray(dip_dcm_chunk))
        energy_chunks.append(np.asarray(energy_chunk))

        processed = end
        current_time = time.time()
        
        # Print progress periodically
        if (current_time - last_print_time >= print_interval) or (batch_idx == n_batches - 1):
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining)
            
            print(f"Processed {processed}/{total} frames ({processed / total:.1%}) | "
                  f"Rate: {rate:.1f} frames/s | "
                  f"ETA: {eta.strftime('%H:%M:%S')}")
            last_print_time = current_time

    dip_phys = np.concatenate(dip_phys_chunks, axis=0)
    dip_dcm = np.concatenate(dip_dcm_chunks, axis=0)
    energies = np.concatenate(energy_chunks, axis=0)

    dipoles_physnet = dip_phys.reshape(n_steps, n_replica, 3)
    dipoles_dcmnet = dip_dcm.reshape(n_steps, n_replica, 3)
    energies = energies.reshape(n_steps, n_replica)
    output_path = args.output or args.positions.with_name(args.positions.stem + "_dipoles.npz")
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Average rate: {total / total_time:.1f} frames/s")
    
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

