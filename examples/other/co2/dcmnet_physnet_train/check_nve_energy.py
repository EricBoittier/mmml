#!/usr/bin/env python3
"""
Check NVE energy conservation from trajectory NPZ files.

For NVE simulations, total energy (kinetic + potential) should be conserved.
This script analyzes energy conservation from saved trajectory data.

Usage:
    python check_nve_energy.py \
        --metadata multi_copy_metadata.npz \
        --positions multi_copy_traj_16x.npz \
        --checkpoint checkpoint_dir \
        [--output-timeseries timeseries.npz]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path to import trainer
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "../../.."
sys.path.insert(0, str(REPO_ROOT.resolve()))

from trainer import JointPhysNetDCMNet, JointPhysNetNonEquivariant  # noqa: E402


def compute_kinetic_energy_from_velocities(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Compute kinetic energy from velocities.
    
    KE = 0.5 * Σ m_i * v_i²
    Conversion: 1 amu * (Å/fs)² = 0.01036427 eV
    
    Parameters
    ----------
    velocities : np.ndarray
        Shape (n_steps, n_replicas, n_atoms, 3) velocities in Å/fs
    masses : np.ndarray
        Shape (n_atoms,) masses in amu
    
    Returns
    -------
    np.ndarray
        Shape (n_steps, n_replicas) kinetic energies in eV
    """
    # velocities: (n_steps, n_replicas, n_atoms, 3)
    # masses: (n_atoms,)
    v_squared = np.sum(velocities**2, axis=-1)  # (n_steps, n_replicas, n_atoms)
    ke_per_atom = 0.5 * masses[None, None, :] * v_squared  # (n_steps, n_replicas, n_atoms)
    ke = np.sum(ke_per_atom, axis=-1) * 0.01036427  # (n_steps, n_replicas) in eV
    return ke


def estimate_kinetic_energy_from_positions(
    positions: np.ndarray, masses: np.ndarray, timestep: float
) -> np.ndarray:
    """
    Estimate kinetic energy from position differences (finite difference).
    
    v ≈ (r(t+dt) - r(t)) / dt
    KE ≈ 0.5 * Σ m_i * v_i²
    
    Parameters
    ----------
    positions : np.ndarray
        Shape (n_steps, n_replicas, n_atoms, 3) positions in Å
    masses : np.ndarray
        Shape (n_atoms,) masses in amu
    timestep : float
        Timestep in fs
    
    Returns
    -------
    np.ndarray
        Shape (n_steps-1, n_replicas) kinetic energies in eV
    """
    # Compute velocities via finite difference
    velocities = np.diff(positions, axis=0) / timestep  # (n_steps-1, n_replicas, n_atoms, 3)
    
    # Compute kinetic energy
    v_squared = np.sum(velocities**2, axis=-1)  # (n_steps-1, n_replicas, n_atoms)
    ke_per_atom = 0.5 * masses[None, None, :] * v_squared  # (n_steps-1, n_replicas, n_atoms)
    ke = np.sum(ke_per_atom, axis=-1) * 0.01036427  # (n_steps-1, n_replicas) in eV
    
    return ke


def build_dense_graph(model_natoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a dense neighbor graph for a single molecule."""
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


def load_checkpoint(checkpoint_dir: Path) -> tuple:
    """Load model and parameters from checkpoint, auto-detecting model type."""
    with open(checkpoint_dir / "best_params.pkl", "rb") as f:
        params = pickle.load(f)
    with open(checkpoint_dir / "model_config.pkl", "rb") as f:
        config = pickle.load(f)
    
    physnet_config = config["physnet_config"]
    mix_coulomb_energy = config.get("mix_coulomb_energy", False)
    
    if "dcmnet_config" in config:
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=config["dcmnet_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )
    elif "noneq_config" in config:
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=config["noneq_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )
    else:
        raise ValueError(
            f"Unknown model type: config must contain either 'dcmnet_config' or 'noneq_config'. "
            f"Found keys: {list(config.keys())}"
        )
    
    return model, params


def compute_forces_batch(
    model,
    params,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    model_natoms: int,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute forces for a batch of positions.
    
    Parameters
    ----------
    model : JointPhysNetDCMNet or JointPhysNetNonEquivariant
        The model
    params : dict
        Model parameters
    positions : np.ndarray
        Shape (n_frames, n_atoms, 3) positions
    atomic_numbers : np.ndarray
        Shape (n_atoms,) atomic numbers
    model_natoms : int
        Model's natoms parameter (for padding)
    batch_size : int
        Batch size for processing
    
    Returns
    -------
    np.ndarray
        Shape (n_frames, n_atoms, 3) forces in eV/Å
    """
    n_frames, n_atoms, _ = positions.shape
    
    # Prepare static inputs
    atomic_numbers_pad = np.zeros((model_natoms,), dtype=np.int32)
    atomic_numbers_pad[:n_atoms] = atomic_numbers.astype(np.int32)
    atom_mask = np.zeros((model_natoms,), dtype=np.float32)
    atom_mask[:n_atoms] = 1.0
    
    # Build graph
    dst_idx_np, src_idx_np = build_dense_graph(model_natoms)
    dst_idx = jnp.array(dst_idx_np)
    src_idx = jnp.array(src_idx_np)
    batch_segments = jnp.zeros((model_natoms,), dtype=jnp.int32)
    batch_mask = jnp.ones((len(dst_idx_np),), dtype=jnp.float32)
    atom_mask_jax = jnp.array(atom_mask)
    atomic_numbers_pad_jax = jnp.array(atomic_numbers_pad)
    
    @jax.jit
    def single_forward(pos_pad: jnp.ndarray):
        output = model.apply(
            params,
            atomic_numbers=atomic_numbers_pad_jax,
            positions=pos_pad,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask_jax,
        )
        forces = output["forces"][:n_atoms]  # Remove padding
        return forces
    
    forward_chunk = jax.jit(lambda pos_chunk: jax.vmap(single_forward)(pos_chunk))
    
    # Pad positions
    positions_pad = np.zeros((n_frames, model_natoms, 3), dtype=np.float32)
    positions_pad[:, :n_atoms, :] = positions.astype(np.float32)
    
    # Process in batches
    all_forces = []
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        pos_chunk = jnp.asarray(positions_pad[start:end], dtype=jnp.float32)
        forces_chunk = forward_chunk(pos_chunk)
        all_forces.append(np.asarray(forces_chunk))
    
    return np.concatenate(all_forces, axis=0)


def check_energy_conservation(
    metadata_path: Path,
    positions_path: Path | None = None,
    timestep: float | None = None,
    checkpoint_path: Path | None = None,
    output_timeseries: Path | None = None,
    force_batch_size: int = 32,
) -> dict:
    """
    Check NVE energy conservation from saved trajectory data.
    
    Parameters
    ----------
    metadata_path : Path
        Path to metadata NPZ file
    positions_path : Path, optional
        Path to positions NPZ file (for kinetic energy estimation)
    timestep : float, optional
        Timestep in fs (required if estimating KE from positions)
    
    Returns
    -------
    dict
        Dictionary with energy conservation statistics
    """
    print("="*70)
    print("NVE ENERGY CONSERVATION CHECK")
    print("="*70)
    
    # Load metadata
    meta = np.load(metadata_path)
    
    ensemble = meta.get("multi_ensemble", "unknown")
    if ensemble != "nve":
        print(f"\n⚠️  Warning: Ensemble is '{ensemble}', not 'nve'")
        print("   Energy conservation check is most meaningful for NVE simulations.")
    
    # Get potential energies
    if "energies" not in meta:
        raise KeyError("'energies' not found in metadata file")
    
    potential_energies = np.asarray(meta["energies"])  # (n_steps, n_replicas)
    n_steps, n_replicas = potential_energies.shape
    
    print(f"\nTrajectory info:")
    print(f"  Steps: {n_steps}")
    print(f"  Replicas: {n_replicas}")
    print(f"  Ensemble: {ensemble}")
    
    # Get masses
    atomic_numbers = np.asarray(meta["atomic_numbers"], dtype=np.int32)
    masses = np.asarray(atomic_numbers, dtype=np.float32)
    # Convert atomic numbers to masses (approximate)
    from ase.data import atomic_masses
    masses = np.array([atomic_masses[int(z)] for z in atomic_numbers])
    
    # Get timestep
    if timestep is None:
        timestep = meta.get("timestep_fs", None)
        if timestep is None:
            raise ValueError("Timestep not found in metadata and not provided")
    
    print(f"  Timestep: {timestep} fs")
    
    # Compute kinetic energy
    kinetic_energies = None
    
    if "velocities" in meta:
        # Use saved velocities if available
        print("\nUsing saved velocities for kinetic energy...")
        velocities = np.asarray(meta["velocities"])
        kinetic_energies = compute_kinetic_energy_from_velocities(velocities, masses)
    elif positions_path and positions_path.exists():
        # Estimate from positions
        print("\nEstimating kinetic energy from position differences...")
        positions_npz = np.load(positions_path)
        positions = positions_npz["positions"]
        positions_npz.close()
        
        if positions.shape[0] != n_steps:
            print(f"  Warning: Positions have {positions.shape[0]} steps, energies have {n_steps}")
            # Align them
            min_steps = min(positions.shape[0], n_steps)
            positions = positions[:min_steps]
            potential_energies = potential_energies[:min_steps]
            n_steps = min_steps
        
        kinetic_energies = estimate_kinetic_energy_from_positions(positions, masses, timestep)
        # Align with potential energies (KE has one fewer step)
        potential_energies = potential_energies[1:]
        n_steps = n_steps - 1
    else:
        print("\n⚠️  Warning: Cannot compute kinetic energy")
        print("   No velocities saved and no positions file provided")
        print("   Only potential energy conservation will be checked")
    
    # Load positions if needed for forces
    positions = None
    if positions_path and positions_path.exists():
        positions_npz = np.load(positions_path)
        positions = positions_npz["positions"]
        positions_npz.close()
        
        # Align positions with energies
        if positions.shape[0] != n_steps:
            min_steps = min(positions.shape[0], n_steps)
            positions = positions[:min_steps]
            if kinetic_energies is not None:
                # Already aligned above
                pass
            else:
                potential_energies = potential_energies[:min_steps]
                n_steps = min_steps
    
    # Compute total energy
    if kinetic_energies is not None:
        total_energies = potential_energies + kinetic_energies  # (n_steps, n_replicas)
    else:
        total_energies = potential_energies
        print("\n⚠️  Only checking potential energy (kinetic energy not available)")
    
    # Compute forces if checkpoint provided
    max_forces = None
    if checkpoint_path and checkpoint_path.exists() and positions is not None:
        print("\nComputing forces from checkpoint...")
        model, model_params = load_checkpoint(checkpoint_path)
        model_natoms = model.physnet_config["natoms"]
        
        # Reshape positions: (n_steps, n_replicas, n_atoms, 3) -> (n_steps * n_replicas, n_atoms, 3)
        positions_flat = positions.reshape(-1, positions.shape[2], 3)
        
        # Compute forces
        forces_flat = compute_forces_batch(
            model, model_params, positions_flat, atomic_numbers, model_natoms, batch_size=force_batch_size
        )
        
        # Reshape back: (n_steps * n_replicas, n_atoms, 3) -> (n_steps, n_replicas, n_atoms, 3)
        forces = forces_flat.reshape(n_steps, n_replicas, positions.shape[2], 3)
        
        # Compute max force magnitude per frame/replica
        force_magnitudes = np.linalg.norm(forces, axis=-1)  # (n_steps, n_replicas, n_atoms)
        max_forces = np.max(force_magnitudes, axis=-1)  # (n_steps, n_replicas)
        
        print(f"  Forces computed: shape {forces.shape}")
        print(f"  Max force range: [{max_forces.min():.6f}, {max_forces.max():.6f}] eV/Å")
    
    # Initialize temperature array if kinetic energies are available
    temperatures_all = None
    if kinetic_energies is not None:
        kB = 8.617333262e-5  # eV/K
        n_atoms = len(masses)
        temperatures_all = 2 * kinetic_energies / (3 * n_atoms * kB)  # (n_steps, n_replicas)
    
    # Analyze energy conservation
    results = {}
    
    print(f"\n{'='*70}")
    print("ENERGY CONSERVATION ANALYSIS")
    print(f"{'='*70}")
    
    # Per replica analysis
    for replica in range(n_replicas):
        print(f"\nReplica {replica}:")
        e_tot = total_energies[:, replica]
        
        e_mean = np.mean(e_tot)
        e_std = np.std(e_tot)
        e_min = np.min(e_tot)
        e_max = np.max(e_tot)
        e_drift = e_tot[-1] - e_tot[0]
        e_drift_pct = 100 * e_drift / e_tot[0] if e_tot[0] != 0 else 0
        
        # Relative energy drift (normalized by initial energy)
        e_drift_relative = e_drift / np.abs(e_tot[0]) if e_tot[0] != 0 else 0
        
        print(f"  Total energy:")
        print(f"    Mean:     {e_mean:.8f} eV")
        print(f"    Std dev:  {e_std:.8f} eV")
        print(f"    Range:    [{e_min:.8f}, {e_max:.8f}] eV")
        print(f"    Drift:    {e_drift:.8f} eV ({e_drift_pct:.6f}%)")
        print(f"    Rel drift: {e_drift_relative:.2e}")
        
        # Energy conservation quality
        if e_std < 1e-6:
            quality = "Excellent"
        elif e_std < 1e-4:
            quality = "Good"
        elif e_std < 1e-2:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"  Quality:   {quality} (std dev: {e_std:.2e} eV)")
        
        results[f"replica_{replica}"] = {
            "mean": float(e_mean),
            "std": float(e_std),
            "min": float(e_min),
            "max": float(e_max),
            "drift": float(e_drift),
            "drift_pct": float(e_drift_pct),
            "quality": quality,
        }
        
        # Potential and kinetic separately (if available)
        if kinetic_energies is not None:
            e_pot = potential_energies[:, replica]
            e_kin = kinetic_energies[:, replica]
            
            print(f"  Potential energy: mean={np.mean(e_pot):.8f} eV, std={np.std(e_pot):.8f} eV")
            print(f"  Kinetic energy:  mean={np.mean(e_kin):.8f} eV, std={np.std(e_kin):.8f} eV")
            
            # Temperature from kinetic energy
            if temperatures_all is not None:
                temperatures = temperatures_all[:, replica]
                print(f"  Temperature:     mean={np.mean(temperatures):.2f} K, std={np.std(temperatures):.2f} K")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    
    # Average over all replicas
    e_tot_all = total_energies.flatten()
    e_mean_all = np.mean(e_tot_all)
    e_std_all = np.std(e_tot_all)
    
    print(f"All replicas combined:")
    print(f"  Mean total energy: {e_mean_all:.8f} eV")
    print(f"  Std dev:          {e_std_all:.8f} eV")
    print(f"  Relative std:     {e_std_all / np.abs(e_mean_all):.2e}")
    
    # Per-step statistics
    e_tot_per_step = np.mean(total_energies, axis=1)  # Average over replicas
    e_drift_per_step = e_tot_per_step - e_tot_per_step[0]
    max_drift = np.max(np.abs(e_drift_per_step))
    
    print(f"\nPer-step analysis (averaged over replicas):")
    print(f"  Max energy drift: {max_drift:.8f} eV")
    print(f"  Final drift:     {e_drift_per_step[-1]:.8f} eV")
    
    results["overall"] = {
        "mean": float(e_mean_all),
        "std": float(e_std_all),
        "relative_std": float(e_std_all / np.abs(e_mean_all)),
        "max_drift": float(max_drift),
        "final_drift": float(e_drift_per_step[-1]),
    }
    
    # Prepare time series data
    if output_timeseries:
        print(f"\n{'='*70}")
        print("SAVING TIME SERIES")
        print(f"{'='*70}")
        
        # Time array (in fs)
        times = np.arange(n_steps) * timestep
        
        # Prepare data dictionary
        timeseries_data = {
            "time_fs": times,
            "step": np.arange(n_steps),
            "total_energy": total_energies,  # (n_steps, n_replicas)
            "potential_energy": potential_energies,  # (n_steps, n_replicas)
        }
        
        if kinetic_energies is not None:
            timeseries_data["kinetic_energy"] = kinetic_energies  # (n_steps, n_replicas)
            if temperatures_all is not None:
                timeseries_data["temperature"] = temperatures_all  # (n_steps, n_replicas)
        
        if max_forces is not None:
            timeseries_data["max_force"] = max_forces  # (n_steps, n_replicas)
        
        # Save to NPZ
        output_timeseries.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_timeseries, **timeseries_data)
        print(f"✅ Time series saved to {output_timeseries}")
        print(f"   Shape: {n_steps} steps × {n_replicas} replicas")
        print(f"   Keys: {list(timeseries_data.keys())}")
    
    meta.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check NVE energy conservation")
    parser.add_argument("--metadata", type=Path, required=True,
                       help="Metadata NPZ file (contains energies)")
    parser.add_argument("--positions", type=Path, default=None,
                       help="Positions NPZ file (required for KE estimation and forces)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Checkpoint directory (required for force computation)")
    parser.add_argument("--timestep", type=float, default=None,
                       help="Timestep in fs (if not in metadata)")
    parser.add_argument("--output-timeseries", type=Path, default=None,
                       help="Output NPZ file for time series data")
    parser.add_argument("--save-json", type=Path, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--force-batch-size", type=int, default=32,
                       help="Batch size for force computation (reduce if OOM)")
    
    args = parser.parse_args()
    
    results = check_energy_conservation(
        args.metadata,
        args.positions,
        args.timestep,
        args.checkpoint,
        args.output_timeseries,
        args.force_batch_size,
    )
    
    if args.save_json:
        import json
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.save_json}")


if __name__ == "__main__":
    main()

