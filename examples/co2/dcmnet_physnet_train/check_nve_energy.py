#!/usr/bin/env python3
"""
Check NVE energy conservation from trajectory NPZ files.

For NVE simulations, total energy (kinetic + potential) should be conserved.
This script analyzes energy conservation from saved trajectory data.

Usage:
    python check_nve_energy.py \
        --metadata multi_copy_metadata.npz \
        [--positions multi_copy_traj_16x.npz] \
        [--masses masses.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def check_energy_conservation(
    metadata_path: Path,
    positions_path: Path | None = None,
    timestep: float | None = None,
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
    masses = np.asarray(meta["atomic_numbers"], dtype=np.float32)
    # Convert atomic numbers to masses (approximate)
    from ase.data import atomic_masses
    masses = np.array([atomic_masses[int(z)] for z in masses])
    
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
    
    # Compute total energy
    if kinetic_energies is not None:
        total_energies = potential_energies + kinetic_energies  # (n_steps, n_replicas)
    else:
        total_energies = potential_energies
        print("\n⚠️  Only checking potential energy (kinetic energy not available)")
    
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
            # KE = (3N/2) * kB * T
            kB = 8.617333262e-5  # eV/K
            n_atoms = len(masses)
            temperatures = 2 * e_kin / (3 * n_atoms * kB)
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
    
    meta.close()
    
    results["overall"] = {
        "mean": float(e_mean_all),
        "std": float(e_std_all),
        "relative_std": float(e_std_all / np.abs(e_mean_all)),
        "max_drift": float(max_drift),
        "final_drift": float(e_drift_per_step[-1]),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check NVE energy conservation")
    parser.add_argument("--metadata", type=Path, required=True,
                       help="Metadata NPZ file (contains energies)")
    parser.add_argument("--positions", type=Path, default=None,
                       help="Optional: Positions NPZ file (for KE estimation)")
    parser.add_argument("--timestep", type=float, default=None,
                       help="Timestep in fs (if not in metadata)")
    parser.add_argument("--save-json", type=Path, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = check_energy_conservation(
        args.metadata,
        args.positions,
        args.timestep,
    )
    
    if args.save_json:
        import json
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.save_json}")


if __name__ == "__main__":
    main()

